#!/usr/bin/env python3
"""
Calibración de SUPERFICIE 3x3 sobre hoja A4 para UR5 + sensor F/T.

Por qué existe
--------------
`calibration_draw.py` palpa UN solo punto y guarda un único Z. Si la mesa no es
plana, ese Z sirve en el centro pero deja el plumón apretado en unas zonas y
levantado en otras. Aquí se palpan 9 puntos repartidos sobre la hoja y se ajusta
una SUPERFICIE z(x, y): al dibujar se evalúa esa superficie en cada waypoint y
la punta sigue la mesa real en lugar de un plano teórico.

Base: `calibration_draw.py` (detección de contacto por Fz y controladores que ya
funcionan en el robot real) + `calibration_surface_A3_3x3.py` (máquina de
estados multipunto y guardado del mapa).

Secuencia
---------
1. Ir a la postura central con el controlador de trayectoria.
2. Construir la malla 3x3 sobre la hoja y VERIFICAR que los 9 puntos sean
   alcanzables (IK offline) antes de mover nada.
3. Pasar a forward_position_controller y, en cada punto:
   traslado en XY a altura segura -> aproximación rápida -> bias local de Fz ->
   descenso lento -> contacto -> retracción.
4. Ajustar z(x, y) por mínimos cuadrados y guardar:
     surface_calibration_A4.npz  -> modelo + malla, listo para consumir
     surface_calibration_A4.csv  -> las 9 medidas crudas
     surface_calibration_A4.txt  -> resumen legible (residuos en mm)
     ~/pintorV2_ws/calibracion/z_calibrado.txt           -> Z del centro, compatible con el launch actual

Todo es parámetro ROS: no hace falta editar el archivo para cambiar la hoja, el
centro, el umbral de fuerza o las velocidades.

    ros2 run ur5_algoritmos calibration_surface_A4_3x3 --ros-args \
        -p center_x:=-0.759 -p center_y:=-0.085 -p probe_speed:=0.002

Consumo posterior (por ejemplo en el nodo de dibujo):

    from ur5_algoritmos.calibration_surface_A4_3x3 import SuperficieZ
    sup = SuperficieZ.cargar('~/pintorV2_ws/calibracion/surface_calibration_A4.npz')
    z = sup.z(x, y)          # escalar o array
"""

import os
import csv
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState

from ur5_algoritmos.cinematica import *


# =============================================================================
# Modelo de superficie
# =============================================================================
# Identificadores usados en el npz y en el mensaje publicado en /calibration_surface
MODELO_PLANO = 1        # z = c0 + c1*u + c2*v
MODELO_CUADRATICO = 2   # z = c0 + c1*u + c2*v + c3*u^2 + c4*u*v + c5*v^2
#   con u = x - cx, v = y - cy (coordenadas relativas al centro de la hoja:
#   centrar mejora mucho el condicionamiento del ajuste)

NOMBRE_MODELO = {MODELO_PLANO: 'plano', MODELO_CUADRATICO: 'cuadratico'}


def _terminos(modelo, u, v):
    """Matriz de diseño del ajuste (una fila por punto)."""
    u = np.atleast_1d(np.asarray(u, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    unos = np.ones_like(u)

    if modelo == MODELO_PLANO:
        return np.column_stack([unos, u, v])

    return np.column_stack([unos, u, v, u * u, u * v, v * v])


class SuperficieZ:
    """
    Superficie z(x, y) de la mesa, tal como la dejó la calibración.

    Se puede evaluar con el modelo ajustado (suave, extrapola con cuidado) o por
    interpolación bilineal de la malla cruda (`z_bilineal`), útil para comparar.
    """

    def __init__(self, modelo, coef, centro, x_values, y_values, z_grid):
        self.modelo = int(modelo)
        self.coef = np.asarray(coef, dtype=float)
        self.centro = np.asarray(centro, dtype=float)
        self.x_values = np.asarray(x_values, dtype=float)
        self.y_values = np.asarray(y_values, dtype=float)
        self.z_grid = np.asarray(z_grid, dtype=float)

    # -------------------------------------------------------------------------
    @staticmethod
    def cargar(ruta):
        ruta = os.path.expanduser(ruta)
        datos = np.load(ruta)
        return SuperficieZ(
            modelo=int(datos['modelo']),
            coef=datos['coef'],
            centro=datos['centro'],
            x_values=datos['x_values'],
            y_values=datos['y_values'],
            z_grid=datos['z_grid'],
        )

    # -------------------------------------------------------------------------
    def z(self, x, y):
        """Z de la mesa en (x, y). Acepta escalares o arrays."""
        escalar = np.isscalar(x) and np.isscalar(y)

        u = np.atleast_1d(np.asarray(x, dtype=float)) - self.centro[0]
        v = np.atleast_1d(np.asarray(y, dtype=float)) - self.centro[1]

        z = _terminos(self.modelo, u, v) @ self.coef

        return float(z[0]) if escalar else z

    # -------------------------------------------------------------------------
    def z_bilineal(self, x, y):
        """Z por interpolación bilineal de la malla medida (sin extrapolar)."""
        xs, ys = self.x_values, self.y_values

        xc = float(np.clip(x, xs.min(), xs.max()))
        yc = float(np.clip(y, ys.min(), ys.max()))

        ix = int(np.clip(np.searchsorted(xs, xc) - 1, 0, len(xs) - 2))
        iy = int(np.clip(np.searchsorted(ys, yc) - 1, 0, len(ys) - 2))

        tx = (xc - xs[ix]) / (xs[ix + 1] - xs[ix])
        ty = (yc - ys[iy]) / (ys[iy + 1] - ys[iy])

        z00 = self.z_grid[iy, ix]
        z10 = self.z_grid[iy, ix + 1]
        z01 = self.z_grid[iy + 1, ix]
        z11 = self.z_grid[iy + 1, ix + 1]

        return (
            z00 * (1 - tx) * (1 - ty)
            + z10 * tx * (1 - ty)
            + z01 * (1 - tx) * ty
            + z11 * tx * ty
        )

    # -------------------------------------------------------------------------
    def __repr__(self):
        return (
            f"SuperficieZ(modelo={NOMBRE_MODELO.get(self.modelo, '?')}, "
            f"centro={np.round(self.centro, 4)}, "
            f"coef={np.round(self.coef, 6)})"
        )


# =============================================================================
# Estados de la máquina de fases
# =============================================================================
PHASE_INIT = 0        # esperando la primera lectura de /joint_states
PHASE_SETUP = 1       # postura central, malla y verificación de alcance
PHASE_TRAVEL = 2      # traslado en XY a altura de viaje
PHASE_APPROACH = 3    # bajada rápida hasta justo encima de la Z estimada
PHASE_BASELINE = 4    # bias local de Fz, con el robot quieto
PHASE_PROBE = 5       # descenso lento hasta detectar contacto
PHASE_RETRACT = 6     # subida a altura de viaje
PHASE_SAVE = 7        # ajuste y guardado
PHASE_HOME = 8        # vuelta a la postura de dibujo
PHASE_FINAL = 9       # fin
PHASE_BUSY = 10       # bloqueo mientras corre una llamada bloqueante


class UR5SurfaceCalibrationA4(Node):

    def __init__(self):
        super().__init__('ur5_surface_calibration_a4')

        # =====================================================================
        # PARÁMETROS
        # =====================================================================
        # --- Hoja y malla ----------------------------------------------------
        # A4 = 210 x 297 mm. Por defecto en VERTICAL (portrait): 210 mm sobre X
        # del robot y 297 mm sobre Y. En horizontal, con el centro de dibujo
        # actual (-0.759, -0.085), la esquina lejana cae a 0.904 m del eje de la
        # base y queda fuera del alcance del UR5; por eso portrait es el default.
        self.declare_parameter('paper_x', 0.210)
        self.declare_parameter('paper_y', 0.297)
        self.declare_parameter('edge_margin', 0.020)

        # Centro de la hoja. Por defecto el mismo DRAW_CENTER que usa
        # imagen_trajectory.py, para que la superficie cubra lo que se dibuja.
        self.declare_parameter('center_x', -0.759)
        self.declare_parameter('center_y', -0.085)
        # True -> ignora center_x/center_y y usa el XY de la postura central.
        self.declare_parameter('center_from_fk', False)

        # --- Postura articular -----------------------------------------------
        self.declare_parameter('q_center', [3.40, -2.14, -1.42, -1.13, np.pi / 2.0, 0.0])
        self.declare_parameter('q_home', [np.pi, -2.14, -1.34, -1.23, np.pi / 2.0, 0.0])
        # Al terminar, volver a la postura central (el punto desde el que se
        # arrancó, en medio de la hoja) en vez de a q_home. Así se puede
        # encadenar otra corrida o el dibujo sin recolocar el robot a mano.
        self.declare_parameter('volver_al_centro', True)

        # --- Orientación de palpado ------------------------------------------
        # Debe ser la MISMA con la que se dibuja: si se palpa con otra, la punta
        # del plumón toca en otro sitio y el Z medido no sirve.
        # Formato [w, x, y, z]; el default es el que publica move_draw_sub.
        self.declare_parameter('quat_palpado', [0.0, 0.70710678, 0.70710678, 0.0])
        # True -> toma la orientación del FK central forzando w=0 y qz=0,
        # como hace calibration_draw.py.
        self.declare_parameter('orientacion_desde_fk', False)

        # --- Palpado ---------------------------------------------------------
        self.declare_parameter('probe_speed', 0.002)       # [m/s] descenso lento
        self.declare_parameter('approach_speed', 0.020)    # [m/s] bajada rápida
        self.declare_parameter('max_probe_depth', 0.060)   # [m] descenso máximo
        self.declare_parameter('approach_height', 0.010)   # [m] sobre la Z estimada
        self.declare_parameter('travel_height', 0.030)     # [m] altura de traslado
        self.declare_parameter('max_z_lag', 0.002)         # [m] referencia vs. real
        self.declare_parameter('pos_tol', 0.0015)          # [m] tolerancia XY/Z
        self.declare_parameter('probe_repeats', 1)         # palpadas por punto

        # Z estimada para el PRIMER punto. NaN -> usa ~/pintorV2_ws/calibracion/z_calibrado.txt si existe
        # y, si no, la altura de la postura central.
        self.declare_parameter('z_inicial_estimado', float('nan'))
        # Suelo absoluto: si la Z real baja de aquí, se aborta.
        self.declare_parameter('z_min_seguridad', 0.100)

        # --- Fuerza ----------------------------------------------------------
        # calibration_draw.py detecta con Fz < -3 N; aquí es un salto de -3 N
        # respecto al bias local, que tolera la deriva del sensor entre puntos.
        self.declare_parameter('contact_delta_n', -3.0)
        self.declare_parameter('contact_samples', 3)
        # Umbral de la guardia anticolisión durante los movimientos rápidos:
        # más alto que el de palpado para no dispararse con la inercia.
        self.declare_parameter('guard_factor', 1.5)
        # El contacto se declara cuando la fuerza YA subió a contact_delta_n, o
        # sea con el plumón algo comprimido: la Z registrada queda unas décimas
        # de mm por debajo de la mesa. El sesgo es constante, así que no deforma
        # la superficie, sólo la baja entera. Con 0.0 la superficie queda igual
        # de sesgada que calibration_draw.py, que es lo que move_draw_sub ya
        # compensa con su +0.0016 m; si se prefiere corregir aquí, poner ese
        # valor en este parámetro y quitarlo del nodo de dibujo (nunca en los
        # dos sitios a la vez).
        self.declare_parameter('contact_offset', 0.0)
        self.declare_parameter('baseline_samples', 25)
        self.declare_parameter('force_alpha', 0.15)

        # --- Control ---------------------------------------------------------
        self.declare_parameter('K', 2.5)
        self.declare_parameter('dq_max', 0.25)

        # --- Controladores ----------------------------------------------------
        self.declare_parameter('controlador_trayectoria', 'scaled_joint_trajectory_controller')
        self.declare_parameter('controlador_posicion', 'forward_position_controller')

        # --- Salidas ----------------------------------------------------------
        self.declare_parameter('output_prefix', '~/pintorV2_ws/calibracion/surface_calibration_A4')
        self.declare_parameter('z_file', '~/pintorV2_ws/calibracion/z_calibrado.txt')
        self.declare_parameter('fit_model', 'auto')     # auto | plano | cuadratico
        self.declare_parameter('skip_failed', True)     # seguir si un punto falla
        self.declare_parameter('verificar_alcance', True)

        P = lambda n: self.get_parameter(n).value

        self.paper_x = float(P('paper_x'))
        self.paper_y = float(P('paper_y'))
        self.edge_margin = float(P('edge_margin'))
        self.center_x = float(P('center_x'))
        self.center_y = float(P('center_y'))
        self.center_from_fk = bool(P('center_from_fk'))

        self.q_center = np.array(P('q_center'), dtype=float)
        self.q_home = np.array(P('q_home'), dtype=float)
        self.volver_al_centro = bool(P('volver_al_centro'))

        self.quat_palpado = np.array(P('quat_palpado'), dtype=float)
        self.quat_palpado = self.quat_palpado / (np.linalg.norm(self.quat_palpado) + 1e-12)
        self.orientacion_desde_fk = bool(P('orientacion_desde_fk'))

        self.probe_speed = float(P('probe_speed'))
        self.approach_speed = float(P('approach_speed'))
        self.max_probe_depth = float(P('max_probe_depth'))
        self.approach_height = float(P('approach_height'))
        self.travel_height = float(P('travel_height'))
        self.max_z_lag = float(P('max_z_lag'))
        self.pos_tol = float(P('pos_tol'))
        self.probe_repeats = max(1, int(P('probe_repeats')))
        self.z_inicial_estimado = float(P('z_inicial_estimado'))
        self.z_min_seguridad = float(P('z_min_seguridad'))

        self.contact_delta_n = float(P('contact_delta_n'))
        self.contact_samples = int(P('contact_samples'))
        self.guard_factor = float(P('guard_factor'))
        self.contact_offset = float(P('contact_offset'))
        self.baseline_samples = int(P('baseline_samples'))
        self.force_alpha = float(P('force_alpha'))

        self.K = float(P('K'))
        self.dq_max = float(P('dq_max'))

        self.ctrl_traj = str(P('controlador_trayectoria'))
        self.ctrl_pos = str(P('controlador_posicion'))

        prefijo = os.path.expanduser(str(P('output_prefix')))
        self.npz_path = prefijo + '.npz'
        self.csv_path = prefijo + '.csv'
        self.txt_path = prefijo + '.txt'
        self.z_file = os.path.expanduser(str(P('z_file')))

        # La carpeta de salida puede no existir todavía (workspace recién
        # clonado, o output_prefix apuntando a otro sitio): se crea aquí, al
        # arrancar, y no a mitad del palpado cuando ya hay datos que perder.
        for carpeta in {os.path.dirname(prefijo), os.path.dirname(self.z_file)}:
            if carpeta:
                os.makedirs(carpeta, exist_ok=True)

        self.fit_model = str(P('fit_model')).lower()
        self.skip_failed = bool(P('skip_failed'))
        self.verificar_alcance = bool(P('verificar_alcance'))

        self.dt = 1.0 / 50.0

        # =====================================================================
        # ROS
        # =====================================================================
        self.client_group = ReentrantCallbackGroup()
        self.timer_group = MutuallyExclusiveCallbackGroup()

        self.position_pub = self.create_publisher(
            Float64MultiArray, f'/{self.ctrl_pos}/commands', 10)

        # Compatibilidad con calibration_draw.py: Z del centro de la hoja.
        self.calib_z_pub = self.create_publisher(Float64, '/calibration_z', 10)

        # Superficie completa: [modelo, cx, cy, coef...]
        self.calib_surface_pub = self.create_publisher(
            Float64MultiArray, '/calibration_surface', 10)

        self.action_client = ActionClient(
            self, FollowJointTrajectory,
            f'/{self.ctrl_traj}/follow_joint_trajectory',
            callback_group=self.client_group)

        self.switch_ctrl_client = self.create_client(
            SwitchController, '/controller_manager/switch_controller',
            callback_group=self.client_group)

        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)

        self.force_sub = self.create_subscription(
            WrenchStamped, '/force_torque_sensor_broadcaster/wrench',
            self.force_cb, 10)

        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # =====================================================================
        # ESTADO
        # =====================================================================
        # q_actual siempre viene de /joint_states; q_cmd es lo que se manda al
        # forward controller. Cerrar el lazo sobre la posición REAL evita que la
        # referencia se despegue del robot durante el palpado.
        self.q_actual = None
        self.q_cmd = None

        self.fz_raw = None
        self.fz_filt = None
        self.fz_baseline_global = None   # referencia en el aire, para la guardia
        self.baseline_activo = None      # None = detección desarmada
        self.umbral_activo = 0.0
        self.baseline_buffer = []
        self.contact_counter = 0
        self.contact_detected = False

        self.phase = PHASE_INIT
        self.pose_base = None       # pose de referencia (orientación de palpado)
        self.safe_z = None

        self.puntos = []            # malla 3x3
        self.idx = 0                # punto actual
        self.repeticion = 0         # palpada actual dentro del punto
        self.z_repeticiones = []

        self.pose_objetivo = None
        self.z_ref = None
        self.z_inicio_palpado = None
        self.z_estimada = None      # predicción para el siguiente punto

        self.medidas = []
        self.ticks_fase = 0
        self.reintentos_punto = 0
        self.accion_tras_retraccion = 'siguiente'
        self.ultimo_delta_fz = np.nan

        self.timer = self.create_timer(self.dt, self.update, self.timer_group)
        self.get_logger().info(
            "Calibración de superficie A4 3x3 iniciada. Esperando /joint_states...")

    # =========================================================================
    # Callbacks
    # =========================================================================
    def joint_state_cb(self, msg: JointState):
        """Actualiza SIEMPRE la posición articular real."""
        try:
            q = np.array([msg.position[msg.name.index(j)] for j in self.joint_names],
                         dtype=float)
        except ValueError:
            return

        primera = self.q_actual is None
        self.q_actual = q

        if self.q_cmd is None:
            self.q_cmd = q.copy()

        if primera:
            self.get_logger().info(f"Posición real inicial: {np.round(q, 3)}")
            self.phase = PHASE_SETUP

    def force_cb(self, msg: WrenchStamped):
        self.fz_raw = float(msg.wrench.force.z)

        if self.fz_filt is None:
            self.fz_filt = self.fz_raw
        else:
            a = self.force_alpha
            self.fz_filt = self.fz_filt + a * (self.fz_raw - self.fz_filt)

        # La detección se arma por fases: palpado fino durante el descenso
        # lento, guardia anticolisión durante los movimientos rápidos.
        if self.baseline_activo is None:
            return

        delta = self.fz_filt - self.baseline_activo

        if delta <= self.umbral_activo:
            self.contact_counter += 1
        else:
            self.contact_counter = 0

        if self.contact_counter >= self.contact_samples:
            self.contact_detected = True

    # =========================================================================
    # Utilidades ROS
    # =========================================================================
    def switch_my_controllers(self, to_activate, to_deactivate):
        while not self.switch_ctrl_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando /controller_manager/switch_controller...')

        request = SwitchController.Request()
        request.activate_controllers = [to_activate]
        request.deactivate_controllers = [to_deactivate]
        request.strictness = SwitchController.Request.STRICT

        self.get_logger().info(f"Switch: activar={to_activate}  desactivar={to_deactivate}")
        future = self.switch_ctrl_client.call_async(request)

        while rclpy.ok():
            if future.done():
                self.get_logger().info("Switch completado.")
                return future.result()
            time.sleep(0.05)

    def send_trajectory_goal(self, q_target, duration_sec):
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Servidor FollowJointTrajectory no disponible.")
            return False

        goal_msg = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = np.asarray(q_target, dtype=float).tolist()
        point.velocities = [0.0] * 6
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1.0) * 1e9)

        trajectory.points.append(point)
        goal_msg.trajectory = trajectory

        send_future = self.action_client.send_goal_async(goal_msg)
        while rclpy.ok() and not send_future.done():
            time.sleep(0.05)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Goal rechazado.")
            return False

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            time.sleep(0.05)

        self.get_logger().info("Movimiento articular completado.")
        return True

    # =========================================================================
    # Cinemática
    # =========================================================================
    def compute_fk(self, q):
        T = fkine_ur5(q)
        return T, np.asarray(TF2xyzquat(T), dtype=float)

    def pose_actual(self):
        _, x = self.compute_fk(self.q_actual)
        return x

    def pose_deseada(self, x, y, z):
        """Pose de 7 elementos con la orientación de palpado.

        El cuaternión se devuelve con el signo más cercano al actual: q y -q son
        la misma rotación, pero el signo equivocado hace que el error de
        orientación empuje a la muñeca por el camino largo.
        """
        quat = self.pose_base[3:].copy()

        if self.q_actual is not None:
            actual = self.pose_actual()[3:]
            if float(np.dot(quat, actual)) < 0.0:
                quat = -quat

        return np.hstack(([x, y, z], quat))

    def error_xyz(self, xd):
        if self.q_actual is None:
            return np.inf
        return float(np.linalg.norm(np.asarray(xd[:3]) - self.pose_actual()[:3]))

    def command_pose(self, xd):
        """Un paso de control cartesiano hacia xd, publicado al forward controller."""
        if self.q_actual is None:
            return

        dq = compute_dq(q=self.q_actual, xd=xd, K=self.K)
        dq = np.clip(dq, -self.dq_max, self.dq_max)

        self.q_cmd = self.q_actual + dq * self.dt

        msg = Float64MultiArray()
        msg.data = self.q_cmd.tolist()
        self.position_pub.publish(msg)

    def ik_offline(self, xd, q0, iters=600):
        """IK iterativa sin mover el robot: sirve para validar la malla antes."""
        q = np.asarray(q0, dtype=float).copy()

        for _ in range(iters):
            dq = np.clip(compute_dq(q=q, xd=xd, K=1.0), -0.3, 0.3)
            q = q + dq * 0.02 * 50.0

            if self.error_offline(q, xd) < 1e-4:
                break

        return q, self.error_offline(q, xd)

    def error_offline(self, q, xd):
        _, x = self.compute_fk(q)
        return float(np.linalg.norm(np.asarray(xd[:3]) - x[:3]))

    # =========================================================================
    # Malla
    # =========================================================================
    def construir_malla(self, cx, cy):
        """
        Malla 3x3 sobre la hoja, recorrida en serpiente para no cruzar la hoja
        de lado a lado entre punto y punto.

        Índices: ix = columna (X del robot), iy = fila (Y del robot), ambos 0..2.
        """
        half_x = max(0.0, self.paper_x / 2.0 - self.edge_margin)
        half_y = max(0.0, self.paper_y / 2.0 - self.edge_margin)

        dx = [-half_x, 0.0, +half_x]
        dy = [-half_y, 0.0, +half_y]

        puntos = []
        for iy in range(3):
            columnas = range(3) if iy % 2 == 0 else range(2, -1, -1)
            for ix in columnas:
                puntos.append({
                    'ix': ix, 'iy': iy,
                    'x': cx + dx[ix], 'y': cy + dy[iy],
                })

        # Empezar por el extremo más cercano al robot.
        actual = self.pose_actual()[:2]
        d_primero = np.hypot(puntos[0]['x'] - actual[0], puntos[0]['y'] - actual[1])
        d_ultimo = np.hypot(puntos[-1]['x'] - actual[0], puntos[-1]['y'] - actual[1])
        if d_ultimo < d_primero:
            puntos.reverse()

        self.puntos = puntos

        self.get_logger().info(
            f"Malla A4 3x3: {self.paper_x*1000:.0f} x {self.paper_y*1000:.0f} mm, "
            f"margen {self.edge_margin*1000:.0f} mm, centro ({cx:.4f}, {cy:.4f})")
        for i, p in enumerate(puntos, start=1):
            self.get_logger().info(
                f"  P{i} (col {p['ix']}, fila {p['iy']}): "
                f"X={p['x']:.4f}  Y={p['y']:.4f}  r={np.hypot(p['x'], p['y']):.4f} m")

    def verificar_malla(self):
        """
        Comprueba con IK offline que los 9 puntos son alcanzables antes de mover
        nada. Se verifica a la altura de VIAJE y a la de palpado, no a la de la
        postura central: el brazo alcanza más lejos cuanto más cerca está del
        plano del hombro, así que usar una altura que no es la de trabajo
        rechaza puntos perfectamente alcanzables (o al revés).
        """
        z_viaje = self.z_estimada + self.travel_height
        alturas = sorted({round(z_viaje, 4), round(self.z_estimada, 4)})

        self.get_logger().info(
            f"Verificando alcance a Z = {', '.join(f'{z:.4f}' for z in alturas)} m")

        malos = []
        q_ref = self.q_actual.copy()

        for i, p in enumerate(self.puntos, start=1):
            for z in alturas:
                xd = self.pose_deseada(p['x'], p['y'], z)
                q_sol, err = self.ik_offline(xd, q_ref)

                if err < 1e-3:
                    q_ref = q_sol      # arrancar el siguiente desde el anterior
                else:
                    malos.append((i, p, err))
                    break

        if malos:
            self.get_logger().error("Puntos FUERA DE ALCANCE a la altura de trabajo:")
            for i, p, err in malos:
                self.get_logger().error(
                    f"  P{i}: X={p['x']:.4f} Y={p['y']:.4f} "
                    f"r={np.hypot(p['x'], p['y']):.4f} m  error residual={err*1000:.1f} mm")
            self.get_logger().error(
                f"Altura de viaje usada: {z_viaje:.4f} m "
                "(bajar travel_height suele ganar algo de alcance).")
            self.get_logger().error(
                "Acerca el centro (center_x/center_y), reduce la hoja o gírala "
                "antes de repetir. Nada se ha movido.")
            return False

        self.get_logger().info("Los 9 puntos son alcanzables.")
        return True

    # =========================================================================
    # Gestión de puntos
    # =========================================================================
    def z_estimada_inicial(self):
        """Mejor conjetura de la altura de la mesa antes de la primera palpada."""
        if not np.isnan(self.z_inicial_estimado):
            return self.z_inicial_estimado

        if os.path.exists(self.z_file):
            try:
                with open(self.z_file, 'r') as f:
                    z = float(f.read().strip())
                self.get_logger().info(
                    f"Usando {self.z_file} como Z estimada inicial: {z:.4f} m")
                return z
            except (OSError, ValueError):
                pass

        return self.safe_z

    def preparar_punto(self):
        """Deja el nodo listo para trasladarse al punto actual."""
        p = self.puntos[self.idx]

        z_viaje = self.z_estimada + self.travel_height

        self.pose_objetivo = self.pose_deseada(p['x'], p['y'], z_viaje)

        self.baseline_buffer = []
        self.armar_guardia()
        self.ticks_fase = 0
        self.phase = PHASE_TRAVEL

        self.get_logger().info(
            f"--- P{self.idx + 1}/9 (palpada {self.repeticion + 1}/{self.probe_repeats}) "
            f"X={p['x']:.4f} Y={p['y']:.4f}  Z viaje={z_viaje:.4f} ---")

    def armar_deteccion(self, baseline, umbral):
        self.baseline_activo = float(baseline)
        self.umbral_activo = float(umbral)
        self.contact_counter = 0
        self.contact_detected = False

    def desarmar_deteccion(self):
        self.baseline_activo = None
        self.contact_counter = 0
        self.contact_detected = False

    def armar_guardia(self):
        """Vigilancia anticolisión para los tramos que se recorren rápido."""
        if self.fz_baseline_global is None:
            self.desarmar_deteccion()
            return
        self.armar_deteccion(self.fz_baseline_global,
                             self.contact_delta_n * self.guard_factor)

    def registrar_medida(self, z, delta_fz, ok=True):
        p = self.puntos[self.idx]
        pose = self.pose_actual()

        self.medidas.append({
            'punto': self.idx + 1,
            'ix': p['ix'], 'iy': p['iy'],
            'x_obj': p['x'], 'y_obj': p['y'],
            'x_real': float(pose[0]), 'y_real': float(pose[1]),
            'z': float(z), 'delta_fz': float(delta_fz), 'ok': bool(ok),
        })

    def siguiente_objetivo(self):
        """Avanza repetición/punto y decide la siguiente fase."""
        self.repeticion += 1

        if self.repeticion < self.probe_repeats:
            self.preparar_punto()
            return

        # Cerrar el punto: promediar las palpadas y guardarlo.
        if self.z_repeticiones:
            z_media = float(np.mean(self.z_repeticiones))
            dispersion = (float(np.max(self.z_repeticiones) - np.min(self.z_repeticiones))
                          if len(self.z_repeticiones) > 1 else 0.0)
            self.registrar_medida(z_media, self.ultimo_delta_fz, ok=True)

            self.z_estimada = z_media
            self.get_logger().info(
                f"P{self.idx + 1}: Z = {z_media:.5f} m"
                + (f" (dispersión {dispersion*1000:.2f} mm)" if dispersion else ""))
        else:
            self.registrar_medida(np.nan, np.nan, ok=False)
            self.get_logger().warn(f"P{self.idx + 1}: sin medida válida.")

        self.z_repeticiones = []
        self.repeticion = 0
        self.reintentos_punto = 0
        self.idx += 1

        if self.idx >= len(self.puntos):
            self.phase = PHASE_SAVE
        else:
            self.preparar_punto()

    def contacto_inesperado(self, contexto):
        """
        Saltó la guardia fuera del descenso lento: la mesa estaba más arriba de
        lo estimado. Se sube, se corrige la estimación con la Z del roce y se
        repite el punto; ese Z no se registra porque un toque a velocidad de
        traslado no es una medida fiable.
        """
        z_toque = float(self.pose_actual()[2])
        self.desarmar_deteccion()
        self.reintentos_punto += 1

        self.get_logger().warn(
            f"P{self.idx + 1}: contacto {contexto} a Z={z_toque:.5f} m. "
            f"Corrigiendo la Z estimada (intento {self.reintentos_punto}).")

        self.z_estimada = z_toque

        p = self.puntos[self.idx]
        self.pose_objetivo = self.pose_deseada(
            p['x'], p['y'], z_toque + self.travel_height)
        self.ticks_fase = 0

        if self.reintentos_punto > 2:
            self.get_logger().error(
                f"P{self.idx + 1}: demasiados contactos en movimiento rápido. "
                "Revisa approach_speed o guard_factor.")
            self.accion_tras_retraccion = 'siguiente'
        else:
            self.accion_tras_retraccion = 'reintentar'

        self.phase = PHASE_RETRACT

    def abortar_punto(self, motivo):
        self.get_logger().error(f"P{self.idx + 1}: {motivo}")

        if not self.skip_failed:
            self.get_logger().error("skip_failed=False: abortando la calibración.")
            self.phase = PHASE_HOME
            return

        # Subir y pasar al siguiente punto sin registrar esta palpada.
        p = self.puntos[self.idx]
        self.desarmar_deteccion()
        self.pose_objetivo = self.pose_deseada(
            p['x'], p['y'], self.z_estimada + self.travel_height)
        self.accion_tras_retraccion = 'siguiente'
        self.ticks_fase = 0
        self.phase = PHASE_RETRACT

    # =========================================================================
    # Ajuste y guardado
    # =========================================================================
    def ajustar_superficie(self, validas):
        """
        Mínimos cuadrados sobre las medidas válidas.

        Con 9 puntos y 6 términos el cuadrático queda sobredeterminado: absorbe
        la comba de la mesa sin seguir el ruido de cada palpada. Si hay menos de
        6 puntos se cae a plano, que sólo necesita 3.
        """
        u = np.array([m['x_obj'] for m in validas]) - self.centro[0]
        v = np.array([m['y_obj'] for m in validas]) - self.centro[1]
        z = np.array([m['z'] for m in validas])

        if self.fit_model == 'plano':
            modelo = MODELO_PLANO
        elif self.fit_model == 'cuadratico':
            modelo = MODELO_CUADRATICO
        else:
            modelo = MODELO_CUADRATICO if len(validas) >= 6 else MODELO_PLANO

        if modelo == MODELO_CUADRATICO and len(validas) < 6:
            self.get_logger().warn(
                f"Sólo {len(validas)} puntos válidos: se ajusta un plano.")
            modelo = MODELO_PLANO

        A = _terminos(modelo, u, v)
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        residuos = z - A @ coef

        return modelo, coef, residuos

    def guardar_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['punto', 'ix', 'iy', 'x_obj', 'y_obj',
                        'x_real', 'y_real', 'z_contacto', 'delta_fz', 'ok'])
            for m in self.medidas:
                w.writerow([m['punto'], m['ix'], m['iy'],
                            f"{m['x_obj']:.6f}", f"{m['y_obj']:.6f}",
                            f"{m['x_real']:.6f}", f"{m['y_real']:.6f}",
                            f"{m['z']:.6f}", f"{m['delta_fz']:.3f}",
                            int(m['ok'])])

    def guardar(self):
        # El CSV se escribe antes que nada: si el ajuste falla, al menos quedan
        # las medidas crudas para entender por qué.
        self.guardar_csv()

        validas = [m for m in self.medidas if m['ok'] and np.isfinite(m['z'])]

        if len(validas) < 3:
            raise RuntimeError(
                f"Sólo {len(validas)} puntos válidos: no alcanza ni para un plano. "
                f"Revisa {self.csv_path}.")

        modelo, coef, residuos = self.ajustar_superficie(validas)

        rms = float(np.sqrt(np.mean(residuos ** 2)))
        max_abs = float(np.max(np.abs(residuos)))

        # --- Malla regular, indexada por (iy, ix) -----------------------------
        half_x = max(0.0, self.paper_x / 2.0 - self.edge_margin)
        half_y = max(0.0, self.paper_y / 2.0 - self.edge_margin)

        x_values = self.centro[0] + np.array([-half_x, 0.0, +half_x])
        y_values = self.centro[1] + np.array([-half_y, 0.0, +half_y])

        # Los huecos se rellenan con el modelo: así z_grid siempre es utilizable.
        z_grid = np.full((3, 3), np.nan)
        for m in self.medidas:
            if m['ok'] and np.isfinite(m['z']):
                z_grid[m['iy'], m['ix']] = m['z']

        superficie = SuperficieZ(modelo, coef, self.centro, x_values, y_values, z_grid)

        for iy in range(3):
            for ix in range(3):
                if not np.isfinite(z_grid[iy, ix]):
                    z_grid[iy, ix] = superficie.z(x_values[ix], y_values[iy])
        superficie.z_grid = z_grid

        z_centro = superficie.z(self.centro[0], self.centro[1])

        # --- NPZ ---------------------------------------------------------------
        np.savez(
            self.npz_path,
            modelo=np.array(modelo),
            modelo_nombre=np.array(NOMBRE_MODELO[modelo]),
            coef=coef,
            centro=self.centro,
            x_values=x_values,
            y_values=y_values,
            z_grid=z_grid,
            z_centro=np.array(z_centro),
            residuos=residuos,
            rms=np.array(rms),
            max_abs=np.array(max_abs),
            paper=np.array([self.paper_x, self.paper_y]),
            edge_margin=np.array(self.edge_margin),
            quat_palpado=self.pose_base[3:],
        )

        # --- Resumen legible ----------------------------------------------------
        z_mod = np.array([[superficie.z(x, y) for x in x_values] for y in y_values])

        lineas = [
            "Calibración de superficie A4 3x3",
            "=" * 60,
            f"Hoja            : {self.paper_x*1000:.0f} x {self.paper_y*1000:.0f} mm "
            f"(margen {self.edge_margin*1000:.0f} mm)",
            f"Centro          : X={self.centro[0]:.4f}  Y={self.centro[1]:.4f}",
            f"Modelo          : {NOMBRE_MODELO[modelo]}",
            f"Coeficientes    : {np.array2string(coef, precision=6)}",
            f"   z(x,y) = c0 + c1*u + c2*v"
            + (" + c3*u^2 + c4*u*v + c5*v^2" if modelo == MODELO_CUADRATICO else "")
            + "   con u = x - cx, v = y - cy",
            f"Z en el centro  : {z_centro:.5f} m",
            f"Puntos válidos  : {len(validas)}/9",
            f"Residuo RMS     : {rms*1000:.3f} mm",
            f"Residuo máximo  : {max_abs*1000:.3f} mm",
            f"Offset aplicado : {self.contact_offset*1000:.3f} mm "
            "(contact_offset; el contacto se detecta con el plumón ya comprimido)",
            f"Desnivel medido : {(np.nanmax(z_grid) - np.nanmin(z_grid))*1000:.3f} mm "
            "(diferencia entre el punto más alto y el más bajo)",
            "",
            "Z medida por celda [mm] (arriba = Y mayor, izquierda = X menor):",
        ]
        for iy in range(2, -1, -1):
            lineas.append("   " + "  ".join(f"{z_grid[iy, ix]*1000:9.3f}" for ix in range(3)))
        lineas += ["", "Z del modelo por celda [mm]:"]
        for iy in range(2, -1, -1):
            lineas.append("   " + "  ".join(f"{z_mod[iy, ix]*1000:9.3f}" for ix in range(3)))

        with open(self.txt_path, 'w') as f:
            f.write("\n".join(lineas) + "\n")

        for l in lineas:
            self.get_logger().info(l)

        if rms > 0.0005:
            self.get_logger().warn(
                f"El modelo deja {rms*1000:.2f} mm de residuo RMS: la mesa tiene "
                "una forma que una superficie suave no captura (¿un escalón, un "
                "punto mal palpado?). Revisa el CSV antes de dibujar.")

        # --- Compatibilidad con el launch actual --------------------------------
        try:
            with open(self.z_file, 'w') as f:
                f.write(str(z_centro))
        except OSError as e:
            self.get_logger().error(f"No se pudo escribir {self.z_file}: {e}")

        # --- Publicaciones -------------------------------------------------------
        msg_z = Float64()
        msg_z.data = float(z_centro)
        self.calib_z_pub.publish(msg_z)

        msg_s = Float64MultiArray()
        msg_s.data = [float(modelo), float(self.centro[0]), float(self.centro[1])] \
            + [float(c) for c in coef]
        self.calib_surface_pub.publish(msg_s)

        self.get_logger().info(f"Guardado: {self.npz_path}")
        self.get_logger().info(f"Guardado: {self.csv_path}")
        self.get_logger().info(f"Guardado: {self.txt_path}")
        self.get_logger().info(f"Guardado: {self.z_file} ({z_centro:.5f} m)")

    # =========================================================================
    # LOOP PRINCIPAL 50 Hz
    # =========================================================================
    def update(self):

        if self.phase in (PHASE_INIT, PHASE_BUSY):
            return

        self.ticks_fase += 1

        # Suelo absoluto: vale para cualquier fase en la que el robot esté abajo.
        if (self.q_actual is not None
                and self.phase in (PHASE_APPROACH, PHASE_BASELINE, PHASE_PROBE)
                and self.pose_actual()[2] < self.z_min_seguridad):
            self.get_logger().error(
                f"Z real por debajo de z_min_seguridad ({self.z_min_seguridad:.3f} m). "
                "Abortando.")
            self.desarmar_deteccion()
            self.phase = PHASE_HOME
            return

        # ---------------------------------------------------------------------
        # SETUP: postura central, malla, verificación y cambio de controlador
        # ---------------------------------------------------------------------
        if self.phase == PHASE_SETUP:
            self.phase = PHASE_BUSY   # las llamadas de abajo son bloqueantes

            self.switch_my_controllers(self.ctrl_traj, self.ctrl_pos)

            if not self.send_trajectory_goal(self.q_center, duration_sec=5.0):
                self.get_logger().error("No se pudo llegar a la postura central.")
                self.phase = PHASE_FINAL
                return

            time.sleep(0.3)   # dejar que /joint_states refleje la llegada

            if self.q_actual is None:
                self.get_logger().error("Sin lectura de /joint_states.")
                self.phase = PHASE_FINAL
                return

            pose_fk = self.pose_actual()

            # Orientación de palpado
            if self.orientacion_desde_fk:
                quat = pose_fk[3:].copy()
                quat[0] = 0.0   # w
                quat[3] = 0.0   # qz
                quat = quat / (np.linalg.norm(quat) + 1e-12)
            else:
                quat = self.quat_palpado

            self.pose_base = np.hstack((pose_fk[:3], quat))
            self.safe_z = float(pose_fk[2])

            cx = pose_fk[0] if self.center_from_fk else self.center_x
            cy = pose_fk[1] if self.center_from_fk else self.center_y
            self.centro = np.array([cx, cy], dtype=float)

            self.get_logger().info(
                f"Postura central: X={pose_fk[0]:.4f} Y={pose_fk[1]:.4f} Z={self.safe_z:.4f}")
            self.get_logger().info(f"Cuaternión de palpado [w,x,y,z]: {np.round(quat, 4)}")

            self.construir_malla(cx, cy)

            self.z_estimada = self.z_estimada_inicial()
            self.get_logger().info(f"Z estimada inicial: {self.z_estimada:.4f} m")

            if self.verificar_alcance and not self.verificar_malla():
                self.phase = PHASE_FINAL
                return

            # Pasar a control cartesiano y arrancar desde donde está el robot.
            self.switch_my_controllers(self.ctrl_pos, self.ctrl_traj)

            msg = Float64MultiArray()
            msg.data = self.q_actual.tolist()
            self.position_pub.publish(msg)

            # Referencia de Fz en el aire: sirve de guardia anticolisión
            # mientras el robot se traslada y baja rápido entre puntos.
            time.sleep(0.5)
            if self.fz_filt is not None:
                self.fz_baseline_global = float(self.fz_filt)
                self.get_logger().info(
                    f"Fz en el aire: {self.fz_baseline_global:.3f} N "
                    f"(guardia a {self.contact_delta_n * self.guard_factor:.1f} N)")
            else:
                self.get_logger().warn(
                    "Sin datos del sensor de fuerza: la guardia anticolisión "
                    "queda desactivada durante los traslados.")

            self.idx = 0
            self.repeticion = 0
            self.z_repeticiones = []
            self.reintentos_punto = 0

            self.preparar_punto()
            self.get_logger().info("Iniciando palpado multipunto.")
            return

        # ---------------------------------------------------------------------
        # TRAVEL: mover XY a la altura de viaje
        # ---------------------------------------------------------------------
        if self.phase == PHASE_TRAVEL:

            if self.contact_detected:
                self.contacto_inesperado('durante el traslado en XY')
                return

            self.command_pose(self.pose_objetivo)

            if self.error_xyz(self.pose_objetivo) < self.pos_tol:
                # Bajar rápido hasta approach_height sobre la Z estimada.
                self.z_ref = float(self.pose_actual()[2])
                self.ticks_fase = 0
                self.phase = PHASE_APPROACH
                self.get_logger().info(
                    f"P{self.idx + 1}: en posición, bajando a "
                    f"{self.z_estimada + self.approach_height:.4f} m")

            elif self.ticks_fase > int(30.0 / self.dt):
                self.abortar_punto("no se alcanzó el XY en 30 s.")
            return

        # ---------------------------------------------------------------------
        # APPROACH: bajada rápida hasta justo encima de la Z estimada
        # ---------------------------------------------------------------------
        if self.phase == PHASE_APPROACH:

            if self.contact_detected:
                self.contacto_inesperado('durante la aproximación rápida')
                return

            z_meta = self.z_estimada + self.approach_height

            # Rampa de bajada, limitando cuánto puede adelantarse la referencia.
            self.z_ref = max(self.z_ref - self.approach_speed * self.dt, z_meta)
            self.z_ref = max(self.z_ref, self.pose_actual()[2] - self.max_z_lag)

            self.pose_objetivo[2] = self.z_ref
            self.command_pose(self.pose_objetivo)

            if abs(self.pose_actual()[2] - z_meta) < self.pos_tol:
                self.desarmar_deteccion()
                self.baseline_buffer = []
                self.ticks_fase = 0
                self.phase = PHASE_BASELINE
                self.get_logger().info(f"P{self.idx + 1}: midiendo bias de Fz...")

            elif self.ticks_fase > int(30.0 / self.dt):
                self.abortar_punto("no se completó la aproximación en 30 s.")
            return

        # ---------------------------------------------------------------------
        # BASELINE: bias local de Fz con el robot quieto
        # ---------------------------------------------------------------------
        if self.phase == PHASE_BASELINE:
            self.command_pose(self.pose_objetivo)   # mantener la pose

            if self.fz_filt is None:
                if self.ticks_fase > int(10.0 / self.dt):
                    self.abortar_punto("sin datos del sensor de fuerza.")
                return

            self.baseline_buffer.append(self.fz_filt)

            if len(self.baseline_buffer) >= self.baseline_samples:
                baseline = float(np.mean(self.baseline_buffer))

                # El bias se refresca en cada punto: así la deriva del sensor
                # entre palpadas no se confunde con contacto.
                self.fz_baseline_global = baseline
                self.armar_deteccion(baseline, self.contact_delta_n)

                self.z_inicio_palpado = float(self.pose_actual()[2])
                self.z_ref = self.z_inicio_palpado
                self.ticks_fase = 0
                self.phase = PHASE_PROBE

                self.get_logger().info(
                    f"P{self.idx + 1}: bias Fz={baseline:.3f} N. "
                    f"Descendiendo a {self.probe_speed*1000:.1f} mm/s...")
            return

        # ---------------------------------------------------------------------
        # PROBE: descenso lento hasta el contacto
        # ---------------------------------------------------------------------
        if self.phase == PHASE_PROBE:

            if self.contact_detected:
                delta = (self.fz_filt - self.baseline_activo
                         if self.fz_filt is not None else np.nan)
                self.desarmar_deteccion()

                pose = self.pose_actual()
                z_contacto = float(pose[2]) + self.contact_offset

                self.z_repeticiones.append(z_contacto)
                self.ultimo_delta_fz = float(delta)

                self.get_logger().info(
                    f"CONTACTO P{self.idx + 1}: Z={z_contacto:.5f} m  ΔFz={delta:.2f} N")

                p = self.puntos[self.idx]
                self.pose_objetivo = self.pose_deseada(
                    p['x'], p['y'], z_contacto + self.travel_height)
                self.accion_tras_retraccion = 'siguiente'
                self.reintentos_punto = 0
                self.armar_guardia()
                self.ticks_fase = 0
                self.phase = PHASE_RETRACT
                return

            profundidad = self.z_inicio_palpado - self.pose_actual()[2]
            if profundidad > self.max_probe_depth:
                self.desarmar_deteccion()
                self.abortar_punto(
                    f"sin contacto tras {self.max_probe_depth*1000:.0f} mm de descenso.")
                return

            # Rampa lenta. El clamp evita que la referencia se adelante al robot:
            # sin él, al tocar la mesa el plumón acabaría presionando de más.
            self.z_ref -= self.probe_speed * self.dt
            self.z_ref = max(self.z_ref, self.pose_actual()[2] - self.max_z_lag)

            self.pose_objetivo[2] = self.z_ref
            self.command_pose(self.pose_objetivo)
            return

        # ---------------------------------------------------------------------
        # RETRACT: subir y pasar a lo siguiente
        # ---------------------------------------------------------------------
        if self.phase == PHASE_RETRACT:
            self.command_pose(self.pose_objetivo)

            if (self.error_xyz(self.pose_objetivo) < self.pos_tol * 2.0
                    or self.ticks_fase > int(30.0 / self.dt)):

                if self.accion_tras_retraccion == 'reintentar':
                    self.accion_tras_retraccion = 'siguiente'
                    self.preparar_punto()      # vuelve a TRAVEL con la Z corregida
                else:
                    self.siguiente_objetivo()
            return

        # ---------------------------------------------------------------------
        # SAVE
        # ---------------------------------------------------------------------
        if self.phase == PHASE_SAVE:
            self.phase = PHASE_BUSY
            try:
                self.guardar()
                self.get_logger().info("CALIBRACIÓN DE SUPERFICIE A4 COMPLETADA.")
            except Exception as e:
                self.get_logger().error(f"Error guardando la calibración: {e}")

            self.phase = PHASE_HOME
            return

        # ---------------------------------------------------------------------
        # HOME
        # ---------------------------------------------------------------------
        if self.phase == PHASE_HOME:
            self.phase = PHASE_BUSY

            self.switch_my_controllers(self.ctrl_traj, self.ctrl_pos)

            destino = self.q_center if self.volver_al_centro else self.q_home
            nombre = 'centro de la hoja' if self.volver_al_centro else 'q_home'
            self.get_logger().info(f"Volviendo a la postura de inicio ({nombre}).")
            self.send_trajectory_goal(destino, duration_sec=4.0)

            self.phase = PHASE_FINAL
            return

        # ---------------------------------------------------------------------
        # FINAL
        # ---------------------------------------------------------------------
        if self.phase == PHASE_FINAL:
            self.get_logger().info("Nodo de calibración finalizado.")
            self.timer.cancel()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = UR5SurfaceCalibrationA4()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
