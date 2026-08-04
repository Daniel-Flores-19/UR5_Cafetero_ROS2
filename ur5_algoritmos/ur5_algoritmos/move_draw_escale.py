#!/usr/bin/env python3
import rclpy
import csv
from rclpy.node import Node
import numpy as np
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from controller_manager_msgs.srv import SwitchController
import time
from rclpy.executors import MultiThreadedExecutor

from ur5_algoritmos.fk_functions import *
from ur5_algoritmos.ik_functions import *
from ur5_algoritmos.kine_control_functions import *
from ur5_algoritmos.QP_functions import *
from ur5_algoritmos.markers import *


# ============================================================
# Estados de la máquina de fases
# ============================================================
PHASE_INIT       = 0   # esperando posición real del robot
PHASE_GOTO_START = 1   # moviendo al primer punto del dibujo (scaled)
PHASE_SWITCHING  = 2   # cambiando de controller
PHASE_DRAW       = 3   # dibujando
PHASE_UP         = 4   # pen-up entre trazos


# ============================================================
# Tamaños de hoja estándar (ancho x alto, metros, portrait)
# ============================================================
SHEET_SIZES_M = {
    "A4": (0.210, 0.297),
    "A3": (0.297, 0.420),
    "A2": (0.420, 0.594),
    "A1": (0.594, 0.841),
}


def scale_points_to_sheet(
    points,
    sheet: str = "A3",
    letter_fraction: float = 1.0 / 5.0,
    draw_center: np.ndarray = None,
    z_contact: float = None,
) -> list:
    """
    Escala una lista de puntos cartesianos [x, y, z] para que el dibujo
    quepa en el área física correspondiente a ``letter_fraction`` del ancho
    de la hoja ``sheet``, centrado en ``draw_center``.

    La escala se calcula a partir del **bounding box** de los puntos
    recibidos: el lado más largo (en x o y) se ajusta al tamaño objetivo,
    preservando la relación de aspecto.  La coordenada ``z`` de cada punto
    se mantiene intacta salvo que se indique ``z_contact``.

    Parámetros
    ----------
    points : list of [x, y, z]  o  np.ndarray (N, 3)
        Puntos en metros, en cualquier escala o posición.
    sheet : str
        Tamaño de hoja destino: ``"A4"``, ``"A3"``, ``"A2"`` o ``"A1"``.
    letter_fraction : float
        Fracción del **ancho** de la hoja que debe ocupar el lado mayor
        del dibujo.  Ejemplos:
            - ``1/5``  → ≈ 5.9 cm en A3  (letra grande)
            - ``1/8``  → ≈ 3.7 cm en A3  (letra mediana)
            - ``1/12`` → ≈ 2.5 cm en A3  (letra pequeña)
    draw_center : np.ndarray, shape (3,)  o  None
        Centro del plano de escritura en el espacio del robot [x, y, z] m.
        Si es ``None`` se usa el centroide de los puntos de entrada
        manteniendo su posición media en x e y, y su z media.
    z_contact : float  o  None
        Si se proporciona, fuerza todos los puntos de salida a esta altura z.
        Útil para asegurar que todos los puntos estén exactamente en el
        plano de contacto del robot (p. ej. ``0.1505``).
        Si es ``None``, se conserva el z original de cada punto.

    Retorna
    -------
    list of [x, y, z]
        Puntos reescalados y recentrados, en la misma estructura de lista
        que recibe ``self.points`` en el nodo.

    Ejemplos de uso
    ---------------
    # En xy_callback, justo después de construir self.points:
    self.points = scale_points_to_sheet(
        self.points,
        sheet          = "A3",
        letter_fraction = 1.0 / 5.0,
        draw_center    = np.array([-0.7, 0.1, 0.1505]),
        z_contact      = 0.1505,
    )

    # Para una hoja completa (sin fracción), letter_fraction=1 usa todo
    # el ancho usable de la hoja menos márgenes (10 % por lado):
    self.points = scale_points_to_sheet(
        self.points,
        sheet          = "A3",
        letter_fraction = 0.80,   # 80 % del ancho → márgenes 10 % c/u
        draw_center    = np.array([-0.7, 0.1, 0.1505]),
        z_contact      = 0.1505,
    )
    """
    if sheet not in SHEET_SIZES_M:
        raise ValueError(
            f"Hoja '{sheet}' no reconocida. Opciones: {list(SHEET_SIZES_M)}"
        )
    if not (0.0 < letter_fraction <= 1.0):
        raise ValueError("letter_fraction debe estar en (0, 1].")
    if len(points) == 0:
        return points

    pts = np.array(points, dtype=float)   # (N, 3)

    # ------------------------------------------------------------------
    # 1. Tamaño físico objetivo (metros) para el lado mayor del dibujo
    # ------------------------------------------------------------------
    sheet_w_m, _ = SHEET_SIZES_M[sheet]
    target_side   = sheet_w_m * letter_fraction   # ej. 0.297 * 0.2 = 0.0594 m

    # ------------------------------------------------------------------
    # 2. Bounding box en x e y  (z no participa en el escalado 2-D)
    # ------------------------------------------------------------------
    x_vals = pts[:, 0]
    y_vals = pts[:, 1]

    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    y_min, y_max = float(y_vals.min()), float(y_vals.max())

    current_W    = x_max - x_min          # ancho actual del dibujo (m)
    current_H    = y_max - y_min          # alto  actual del dibujo (m)
    current_side = max(current_W, current_H)   # lado mayor

    # ------------------------------------------------------------------
    # 3. Factor de escala uniforme (mantiene relación de aspecto)
    # ------------------------------------------------------------------
    if current_side < 1e-9:
        # Todos los puntos son coincidentes; devolver sin cambios
        return points

    scale = target_side / current_side

    # ------------------------------------------------------------------
    # 4. Centro del bounding box de entrada (para centrar en el destino)
    # ------------------------------------------------------------------
    cx_in = (x_min + x_max) / 2.0
    cy_in = (y_min + y_max) / 2.0

    # ------------------------------------------------------------------
    # 5. Centro de destino en el workspace del robot
    # ------------------------------------------------------------------
    if draw_center is not None:
        cx_out = float(draw_center[0])
        cy_out = float(draw_center[1])
    else:
        # Sin draw_center: mantener el centroide original en x e y
        cx_out = cx_in
        cy_out = cy_in

    # ------------------------------------------------------------------
    # 6. Aplicar transformación:  p_out = (p_in - c_in) * scale + c_out
    # ------------------------------------------------------------------
    pts_scaled = pts.copy()
    pts_scaled[:, 0] = (x_vals - cx_in) * scale + cx_out
    pts_scaled[:, 1] = (y_vals - cy_in) * scale + cy_out

    # Forzar z de contacto si se indicó
    if z_contact is not None:
        pts_scaled[:, 2] = float(z_contact)

    return pts_scaled.tolist()


def leer_csv():
    with open('trayectoria.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # saltar encabezado
        for row in reader:
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])
            puntos.append([x, y, z, 0, 0.70, 0.70, 0])


class UR5ControlNode(Node):

    def __init__(self):
        super().__init__('ur5_kinecontrol_node')

        # ===== PARAMETROS =====
        self.dt   = 1.0 / 50.0
        self.K    = 2
        self.lamb = 0.01

        self.client_group = ReentrantCallbackGroup()
        self.timer_group  = MutuallyExclusiveCallbackGroup()

        # ===== Publishers / Clients =====
        self.position_pub = self.create_publisher(
            Float64MultiArray, '/forward_position_controller/commands', 10)

        self.action_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self.client_group)

        self.switch_ctrl_client = self.create_client(
            SwitchController,
            '/controller_manager/switch_controller',
            callback_group=self.client_group)

        self.marker_pub = self.create_publisher(Marker, 'ee_marker', 10)

        # ===== Suscripción a joint_states para leer posición real =====
        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)

        self.xy_sub = self.create_subscription(
            JointState, '/letter_trajectory/xy', self.xy_callback, 10)

        self.flags_sub = self.create_subscription(
            JointState, '/letter_trajectory/flags', self.flag_callback, 10)

        # ===== Joint names =====
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # ===== Estado =====
        self.q       = None
        self.phase   = PHASE_INIT

        # ===== Marker =====
        self.marker = create_sphere_marker(
            frame="base_link_inertia", ns="ee", marker_id=0,
            scale=0.05, color=(0.0, 0.0, 1.0, 1.0))

        self.archivo = open('posiciones_robot.txt', 'w')
        self.archivo.write('timestamp,x,y,z\n')

        # ===== Trayectoria circular (parámetros de respaldo) =====
        self.t      = 0.0
        self.radius = 0.1
        self.omega  = 0.5
        self.center = np.array([-0.7, 0.1, 0.1505])

        self.points_received = False
        print("a")
        print(self.points_received)

        # ===== Timer principal =====
        self.timer = self.create_timer(self.dt, self.update, self.timer_group)
        self.get_logger().info("NodoOOO iniciado. Esperando joint_states...")

    # =========================
    # Callback joint_states
    # =========================
    def joint_state_cb(self, msg: JointState):
        """Lee posición real solo hasta tener la primera lectura."""
        if self.q is not None:
            return
        try:
            self.q = np.array([
                msg.position[msg.name.index(j)]
                for j in self.joint_names
            ])
            print(msg.position)
            print(self.q)
            self.get_logger().info(f"Posición real leída: {np.round(self.q, 3)}")
            self.phase = 6  # PHASE_GOTO_START
        except ValueError:
            pass

    def xy_callback(self, msg):
        """
        Recibe pares (x, y) del publisher de trayectoria, construye
        self.points y aplica el escalado A3.

        Ajusta las dos constantes de la llamada a scale_points_to_sheet()
        para cambiar el tamaño del dibujo:
            sheet           → tamaño de hoja ("A4", "A3", "A2", "A1")
            letter_fraction → fracción del ancho de la hoja
                              1/5  = letra grande  (~5.9 cm en A3)
                              1/8  = letra mediana (~3.7 cm en A3)
                              1/12 = letra pequeña (~2.5 cm en A3)
        """
        data = msg.data

        self.points = []
        for i in range(0, len(data), 2):
            x = -data[i]
            y = data[i + 1]
            z = 0.1505
            self.points.append([x, y, z])

        # ── Escalado A3 ──────────────────────────────────────────────
        self.points = scale_points_to_sheet(
            self.points,
            sheet           = "A3",
            letter_fraction = 1.0 / 5.0,          # ← ajustar tamaño aquí
            draw_center     = np.array([-0.7, 0.1, 0.1505]),
            z_contact       = 0.1505,
        )
        # ─────────────────────────────────────────────────────────────

        self.points_received = True

    def flag_callback(self, msg):
        self.flags = list(msg.data)
        self.flags_received = True

    # =========================
    # Switch controller
    # =========================
    def switch_my_controllers(self, to_activate, to_deactivate):
        while not self.switch_ctrl_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando servicio SwitchController...')

        request = SwitchController.Request()
        request.activate_controllers   = [to_activate]
        request.deactivate_controllers = [to_deactivate]
        request.strictness = SwitchController.Request.STRICT

        self.get_logger().info(f"Switch: activar={to_activate}  desactivar={to_deactivate}")
        future = self.switch_ctrl_client.call_async(request)

        while rclpy.ok():
            if future.done():
                self.get_logger().info("Switch completado.")
                return future.result()
            time.sleep(0.05)

    # =========================
    # Mover al inicio del dibujo
    # =========================
    def go_to_draw_start(self):
        """Calcula el q inicial del dibujo y mueve allí con scaled."""
        xd_start = circular_trajectory(0.0, self.center, self.radius, self.omega)

        q_start = self.q.copy()
        for _ in range(500):
            dq = compute_dq_qp(
                fkine=fkine_ur5,
                jacobian_func=numerical_jacobian,
                TF2xyzquat=TF2xyzquat,
                q=q_start,
                xd=xd_start,
                K=2.0,
                lamb=self.lamb,
                dt=0.02
            )
            dq = np.clip(dq, -2.0, 2.0)
            q_start = q_start + dq * 0.02
            _, x = self.compute_fk(q_start)
            if np.linalg.norm(pose_error(xd_start, x)) < 0.005:
                break

        self.get_logger().info(f"Moviendo al inicio del círculo: {np.round(q_start, 3)}")
        q_start = np.array([np.pi, -2.14, -1.34, -1.23, np.pi / 2, 0.0])

        if not self.send_trajectory_goal(q_start, duration_sec=8.0):
            self.get_logger().error("Falló el movimiento al inicio del círculo")
            return False

        self.q = q_start
        return True

    # =========================
    # Send trajectory goal
    # =========================
    def send_trajectory_goal(self, q_target, duration_sec):
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Servidor de acción no disponible.")
            return False

        goal_msg   = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions  = q_target.tolist()
        point.velocities = [0.0] * 6
        point.time_from_start.sec     = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)

        trajectory.points.append(point)
        goal_msg.trajectory = trajectory

        send_goal_future = self.action_client.send_goal_async(goal_msg)
        while rclpy.ok():
            if send_goal_future.done():
                break
            time.sleep(0.05)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rechazado")
            return False

        result_future = goal_handle.get_result_async()
        while rclpy.ok():
            if result_future.done():
                self.get_logger().info("Movimiento completado.")
                return True
            time.sleep(0.05)

    # =========================
    # FK
    # =========================
    def compute_fk(self, q):
        T = fkine_ur5(q)
        x = TF2xyzquat(T)
        return T, x

    # =========================
    # LOOP PRINCIPAL  50 Hz
    # =========================
    def update(self):

        # ── FASE 0: esperar lectura real ──────────────────────────
        if self.phase == PHASE_INIT:
            if not self.points_received:
                print("wait")
                return
            self.phase = PHASE_GOTO_START

        # ── FASE 1: ir al inicio del dibujo con scaled ───────────
        if self.phase == PHASE_GOTO_START:
            self.phase = PHASE_SWITCHING
            success = self.go_to_draw_start()
            if not success:
                self.get_logger().error("No se pudo ir al inicio. Abortando.")
                return

            # ── FASE 2: cambiar a forward_position_controller ─────
            self.switch_my_controllers(
                to_activate   = 'forward_position_controller',
                to_deactivate = 'joint_trajectory_controller'
            )
            self.phase = PHASE_DRAW
            print(self.phase)
            self.get_logger().info("¡Iniciando dibujo!")
            self.punto_goal = 0
            return

        # ── FASE 3: dibujo con forward_position_controller ───────
        if self.phase == PHASE_DRAW:

            xd = self.points[self.punto_goal]
            print(xd)

            dq = compute_dq(q=self.q, xd=xd, K=self.K)
            dq_max = 0.25
            dq = np.clip(dq, -dq_max, dq_max)

            self.q = self.q + dq * self.dt

            msg = Float64MultiArray()
            msg.data = self.q.tolist()
            self.position_pub.publish(msg)

            _, x = self.compute_fk(self.q)
            t = self.get_clock().now().to_msg().sec
            self.archivo.write(f"{t},{x[0]},{x[1]},{x[2]}\n")

            marker = set_marker_pose(self.marker, x, self)
            if marker is not None:
                self.marker = marker
                self.marker_pub.publish(self.marker)

            error = np.linalg.norm(pose_error(xd, x))
            print(error)

            if np.linalg.norm(error) < 0.005:
                self.punto_goal += 1
                xd   = self.points[self.punto_goal]
                flag = self.flags[self.punto_goal]
                if flag == 1:
                    self.phase = PHASE_UP

            self.t += self.dt
            error = np.linalg.norm(pose_error(xd, x))

        # ── FASE 4: pen-up entre trazos ───────────────────────────
        if self.phase == PHASE_UP:
            self.switch_my_controllers(
                to_activate   = 'joint_trajectory_controller',
                to_deactivate = 'forward_position_controller'
            )

            # Levantar el plumón
            xd = self.points[self.punto_goal - 1]
            xd[2] = 0.20

            for _ in range(500):
                dq = compute_dq_qp(
                    fkine=fkine_ur5,
                    jacobian_func=numerical_jacobian,
                    TF2xyzquat=TF2xyzquat,
                    q=self.q,
                    xd=xd,
                    K=2.0,
                    lamb=self.lamb,
                    dt=0.02
                )
                dq = np.clip(dq, -2.0, 2.0)
                self.q = self.q + dq * self.dt
                _, x = self.compute_fk(self.q)
                if np.linalg.norm(pose_error(xd, x)) < 0.005:
                    break

            self.send_trajectory_goal(self.q, duration_sec=5.0)

            # Mover al siguiente punto de dibujo (en altura de vuelo)
            xd = self.points[self.punto_goal]
            xd[2] = 0.20

            for _ in range(500):
                dq = compute_dq_qp(
                    fkine=fkine_ur5,
                    jacobian_func=numerical_jacobian,
                    TF2xyzquat=TF2xyzquat,
                    q=self.q,
                    xd=xd,
                    K=2.0,
                    lamb=self.lamb,
                    dt=0.02
                )
                dq = np.clip(dq, -2.0, 2.0)
                self.q = self.q + dq * self.dt
                _, x = self.compute_fk(self.q)
                if np.linalg.norm(pose_error(xd, x)) < 0.005:
                    break

            self.send_trajectory_goal(self.q, duration_sec=5.0)

            self.phase = PHASE_DRAW

            self.switch_my_controllers(
                to_activate   = 'forward_position_controller',
                to_deactivate = 'joint_trajectory_controller'
            )


def main(args=None):
    rclpy.init(args=args)
    node = UR5ControlNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.archivo.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
