#!/usr/bin/env python3
import rclpy
import csv
import os
from rclpy.node import Node
import numpy as np
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool, Float64
from visualization_msgs.msg import Marker
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from controller_manager_msgs.srv import SwitchController  
import time
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32MultiArray, Int8MultiArray, MultiArrayDimension

from ur5_algoritmos.cinematica import *
from ur5_algoritmos.utils.markers import *
from ur5_algoritmos.calibration_surface_A4_4x4 import SuperficieZ


# Estados de la máquina de fases
PHASE_INIT      = 0   # esperando posición real del robot
PHASE_GOTO_START = 1  # moviendo al primer punto del círculo (scaled)
PHASE_SWITCHING  = 2  # cambiando de controller
PHASE_DRAW     = 3  # dibujando el círculo (forward)
PHASE_UP    = 4
PHASE_FINISH = 5 
#puntos = []



class UR5ControlNode(Node):

    def __init__(self):
        super().__init__('ur5_kinecontrol_node')

        # ===== PARAMETROS =====
        self.dt   = 1.0 / 250.0
        self.K    = 8.5
        self.lamb = 0.015

        self.client_group = ReentrantCallbackGroup()
        self.timer_group  = MutuallyExclusiveCallbackGroup()

        # ===== Publishers / Clients =====
        self.position_pub = self.create_publisher(
            Float64MultiArray, '/forward_position_controller/commands', 10)

        self.action_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self.client_group)

        self.switch_ctrl_client = self.create_client(
            SwitchController,
            '/controller_manager/switch_controller',
            callback_group=self.client_group)    # ← mismo grupo para no bloquear

        # ===== Suscripción a joint_states para leer posición real =====
        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)
        
            
        self.xy_sub = self.create_subscription(
            Float32MultiArray, '/letter_trajectory/xy', self.xy_callback, 10)
            
        self.flags_sub = self.create_subscription(
            Int8MultiArray, '/letter_trajectory/flags', self.flag_callback, 10)

        # ===== Joint names =====
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        # ===== Parametros ====
        
        # Declarar el parámetro con un valor por defecto (por si acaso)
        self.declare_parameter('dist_z', 0.16745639390207478)
        
        # Mapa de la mesa medido por calibration_surface_A4_4x4. Si existe, cada
        # punto del dibujo usa la Z de la superficie en su propio XY en vez de
        # una única altura para toda la hoja: así el plumón no queda apretado en
        # las zonas altas ni levantado en las bajas.
        self.declare_parameter('superficie_npz', '~/pintor_ws/src/calibracion/surface_calibration_A4.npz')
        self.declare_parameter('usar_superficie', True)
        
        # Altura a la que se levanta el plumón entre trazos, medida DESDE el
        # papel. NaN conserva el comportamiento anterior (dist_z * 1.06).
        self.declare_parameter('lift_height', float('nan'))

        # Al terminar, volver a la postura desde la que arrancó el nodo (el
        # centro de la hoja, si se colocó ahí antes de lanzarlo) en vez de a la
        # postura fija de siempre. Así se puede encadenar otro dibujo sin
        # recolocar el robot a mano.
        self.declare_parameter('volver_al_inicio', True)

        # Al terminar, el plumón primero SUBE en vertical esta altura sobre el
        # papel y recién entonces viaja; el regreso termina a esta misma altura
        # sobre el punto de inicio. Sin esto el viaje articular desde el último
        # punto (plumón apoyado) rayaba la hoja en diagonal.
        self.declare_parameter('altura_regreso', 0.03)

        # ===== Seguimiento del trazo =====
        # 'velocidad': la referencia recorre cada trazo a velocidad constante
        #   (velocidad_dibujo) con avance + realimentación, integrando con el
        #   tiempo real transcurrido; si el error pasa de error_max_avance la
        #   referencia se detiene hasta que el robot la alcanza, así los
        #   detalles pequeños no se recortan.
        # 'puntos': comportamiento anterior, salta al siguiente waypoint cuando
        #   el error baja de limit_error.
        self.declare_parameter('modo_seguimiento', 'velocidad')
        self.declare_parameter('velocidad_dibujo', 0.02)      # [m/s]
        self.declare_parameter('error_max_avance', 0.0005)    # [m]
        self.declare_parameter('tol_fin_trazo', 0.0003)       # [m]
        self.declare_parameter('limit_error', 0.002)          # [m] modo 'puntos'
        self.declare_parameter('tol_bajada', 0.0003)          # [m] (antes 0.1 mm)
        self.declare_parameter('dt_max', 0.02)                # [s] tope por ciclo
        P = lambda n: self.get_parameter(n).value
        self.modo_seguimiento = str(P('modo_seguimiento'))
        self.velocidad_dibujo = float(P('velocidad_dibujo'))
        self.error_max_avance = float(P('error_max_avance'))
        self.tol_fin_trazo = float(P('tol_fin_trazo'))
        self.tol_bajada = float(P('tol_bajada'))
        self.dt_max = float(P('dt_max'))
        self.trazo = None            # estado del trazo en curso (modo velocidad)
        self._t_prev = None

        # False -> el traslado al inicio del dibujo parte de la posición actual
        # del robot. True -> pasa antes por la postura fija de siempre.
        self.declare_parameter('ir_a_postura_fija', False)
        self.ir_a_postura_fija = self.get_parameter(
            'ir_a_postura_fija').get_parameter_value().bool_value

        # Leer el parámetro que le envió el archivo Launch
        self.dist_z = self.get_parameter('dist_z').get_parameter_value().double_value
        
        self.get_logger().info(f"Altura Z recibida desde calibración: {self.dist_z}")
        
        
        # ===== Estado =====
        self.q       = None          
        self.q_inicio = None         # postura de arranque, para volver al final
        self.phase   = PHASE_INIT

        # ===== Marker =====
        self.marker = create_sphere_marker(
            frame="base_link_inertia", ns="ee", marker_id=0,
            scale=0.05, color=(0.0, 0.0, 1.0, 1.0))

        # Junto a los datos de calibración, y no en el cwd desde el que se lanzó
        # el nodo: así el log siempre aparece en el mismo sitio.
        ruta_log = os.path.expanduser(
            '~/pintor_ws/src/calibracion/posiciones_robot.txt')
        os.makedirs(os.path.dirname(ruta_log), exist_ok=True)
        self.archivo = open(ruta_log, 'w')
        self.archivo.write('timestamp,x,y,z\n')

        self.t  = 0.0
        self.temp = 1
        
        # Distancia del efector a la mesa -------------
        # Correcciones sobre la Z calibrada (positivo = hacia arriba):
        #  offset_contacto : el contacto se detecta con el plumón ya comprimido,
        #                    así que la Z calibrada queda algo por debajo del
        #                    papel. +1.0 mm: a la Z calibrada tal cual, y luego
        #                    con +0.5 mm, el plumón seguía presionando demasiado.
        #  offset_tapa     : si se calibra con la TAPA puesta, la punta queda
        #                    más arriba y hay que bajar (antes -8.0 mm).
        self.declare_parameter('offset_contacto', 0.0004) # 0.0007 plumon grueso
        self.declare_parameter('offset_tapa', 0.0)
        self.offset_contacto = self.get_parameter(
            'offset_contacto').get_parameter_value().double_value
        self.offset_tapa = self.get_parameter(
            'offset_tapa').get_parameter_value().double_value

        self.z_offset = self.offset_contacto + self.offset_tapa
        self.dist_z = self.dist_z + self.z_offset

        self.get_logger().info(
            f"Offset de Z: {self.z_offset*1000:+.1f} mm "
            f"(contacto {self.offset_contacto*1000:+.1f}, tapa {self.offset_tapa*1000:+.1f})")
        
        self.dist_z_up = self.dist_z * 1.06
        
        # ===== Superficie de la mesa =====
        self.superficie = self.cargar_superficie()
        
        lift = self.get_parameter('lift_height').get_parameter_value().double_value
        self.lift_height = (self.dist_z_up - self.dist_z) if np.isnan(lift) else lift
        self.get_logger().info(
            f"Levantamiento entre trazos: {self.lift_height*1000:.1f} mm sobre el papel")
        
        
        self.cont = 0
        self.limit_error = float(self.get_parameter('limit_error').value)
        self.up_subphase = 0
        self. flag_calibration = 0
        
        
        self.points_received = False
        self.points = []
        self.flags = []

        # ===== Timer principal =====
        self.timer = self.create_timer(self.dt, self.update, self.timer_group)
        self.get_logger().info("Nodo iniciado. Esperando joint_states y puntos de dibujo ...")

    # =========================
    # Altura del papel
    # =========================
    def cargar_superficie(self):
        """Carga el mapa de la mesa; devuelve None si no hay o no se puede leer."""
        if not self.get_parameter('usar_superficie').get_parameter_value().bool_value:
            self.get_logger().info("usar_superficie=False: Z plana en toda la hoja.")
            return None

        ruta = os.path.expanduser(
            self.get_parameter('superficie_npz').get_parameter_value().string_value)

        if not os.path.exists(ruta):
            self.get_logger().warn(
                f"No existe {ruta}: se dibuja con Z plana ({self.dist_z:.4f} m). "
                "Corre calibration_surface_A4_4x4 para medir la mesa.")
            return None

        try:
            superficie = SuperficieZ.cargar(ruta)
        except Exception as e:
            self.get_logger().error(
                f"No se pudo leer {ruta} ({e}): se dibuja con Z plana.")
            return None

        self.get_logger().info(f"Superficie de la mesa cargada de {ruta}: {superficie}")
        return superficie

    def z_papel(self, x, y):
        """Z a la que el plumón toca el papel en (x, y)."""
        if self.superficie is None:
            return self.dist_z
        return float(self.superficie.z(x, y)) + self.z_offset

    def z_arriba(self, x, y):
        """Z de tránsito: la misma altura sobre el papel en cualquier punto."""
        if self.superficie is None:
            return self.dist_z_up
        return self.z_papel(x, y) + self.lift_height

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
            #print(msg.position)
            #print(self.q)
            # self.q la va mutando el lazo de control, así que la postura de
            # arranque se guarda aparte para poder volver a ella al final.
            self.q_inicio = self.q.copy()
            self.get_logger().info(f"Posición real leída: {np.round(self.q, 3)}")
            self.phase = 6 #PHASE_GOTO_START   # arrancar fase 1
        except ValueError:
            pass
           
        
        
    def xy_callback(self, msg):

        # Los publicadores repiten la trayectoria varias veces. Una vez empezado
        # el dibujo se ignoran esas repeticiones: este callback corre en otro
        # hilo (MultiThreadedExecutor) y reconstruir self.points mientras el
        # lazo de dibujo la recorre le hacía ver una lista vacía o a medias y
        # dar el dibujo por terminado tras el primer trazo.
        if self.temp == 0:
            return

        data = msg.data
        # Se construye aparte y se asigna de una vez: el lazo nunca ve una
        # lista a medio llenar.
        points = []

        for i in range(0, len(data), 2):

            x = data[i]
            y = data[i+1]
            z = self.z_papel(x, y)
            points.append([x, y, z, 0, 0.70 , 0.70, 0])

        self.points = points
        self.points_received = True

        if self.superficie is not None:
            zs = [p[2] for p in self.points]
            self.get_logger().info(
                f"{len(self.points)} puntos: Z entre {min(zs)*1000:.2f} y "
                f"{max(zs)*1000:.2f} mm (recorrido {(max(zs)-min(zs))*1000:.2f} mm "
                "siguiendo la mesa)")
        
        
    def flag_callback(self, msg):

        if self.temp == 0:   # dibujo en curso: mismas razones que xy_callback
            return

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
    # Mover al inicio del círculo (bloqueante, solo una vez)
    # =========================
    def go_to_draw_start(self):
        """Calcula el q inicial el dibujo y mueve allí con scaled."""
        # Punto cartesiano inicial del círculo (t=0)
        #xd_start = circular_trajectory(0.0, self.center, self.radius, self.omega)


        if self.ir_a_postura_fija:
            # Comportamiento antiguo: pasar antes por una postura fija. Esa
            # postura está al otro lado de la base (X < 0), así que sólo sirve
            # si la hoja está ahí.
            q_start = np.array([np.pi, -2.14, -1.34, -1.23, np.pi/2, 0.0])
            self.send_trajectory_goal(q_start, duration_sec = 5.0)
        else:
            # Partir de donde está el robot: la IK arranca de la postura real y
            # converge a la rama cercana, sin cruzar el brazo al otro lado.
            q_start = self.q_inicio.copy()
            _, x_aqui = self.compute_fk(q_start)
            self.get_logger().info(
                f"Partiendo de la posición actual: X={x_aqui[0]:.4f} "
                f"Y={x_aqui[1]:.4f} Z={x_aqui[2]:.4f}")

        # Llegar SOBRE el primer punto, con el plumón levantado: el lazo de
        # dibujo baja luego en vertical hasta el papel. Así el traslado
        # articular no arrastra la punta por la hoja.
        xd_start = list(self.points[0])
        xd_start[2] = self.z_arriba(xd_start[0], xd_start[1])

        convergio = False
        #Posicion Inicial del propio Dibujo
        for _ in range(500):   # iterar hasta converger
            dq = compute_dq_qp(
                fkine=fkine_ur5,
                jacobian_func=numerical_jacobian,
                TF2xyzquat=TF2xyzquat,
                q=q_start,
                xd=xd_start,
                K=2.0,
                lamb=self.lamb,
                dt=0.008
            )
            dq = np.clip(dq, -2.0, 2.0)
            q_start = q_start + dq * 0.02
            _, x = self.compute_fk(q_start)
            if np.linalg.norm(pose_error(xd_start, x)) < 0.005:
                print("posicion inicial de dibujo")
                convergio = True
                break

        if not convergio:
            # Mandar un q que no llega al punto es justo lo que hacía que el
            # robot se fuera a otro lado: mejor no moverse.
            self.get_logger().error(
                f"La IK no llegó al inicio del dibujo X={xd_start[0]:.4f} "
                f"Y={xd_start[1]:.4f} (error {np.linalg.norm(pose_error(xd_start, x))*1000:.1f} mm). "
                "No se mueve el robot.")
            return False

        self.get_logger().info(f"Moviendo al inicio del dibujo: {np.round(q_start, 3)}")

        # Enviar con scaled (bloqueante)
        if not self.send_trajectory_goal(q_start, duration_sec=3.0):
            self.get_logger().error("Falló el movimiento al inicio del círculo")
            return False

        # Actualizar q interno con la posición de llegada
        self.q = q_start
        return True

    # =========================
    # Send trajectory goal (bloqueante)
    # =========================
    def send_trajectory_goal(self, q_target, duration_sec):
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Servidor de acción no disponible.")
            return False

        goal_msg  = FollowJointTrajectory.Goal()
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
    # Dibujo a velocidad constante
    # =========================
    def _iniciar_trazo(self, i0):
        """Prepara el trazo que empieza en i0 (hasta el siguiente flag == 1)."""
        i1 = i0 + 1
        while i1 < len(self.points) and self.flags[i1] != 1:
            i1 += 1
        pts = np.array(self.points[i0:i1], dtype=float)
        seg = np.linalg.norm(np.diff(pts[:, :3], axis=0), axis=1) if len(pts) > 1 else np.zeros(0)
        self.trazo = {'i0': i0, 'i1': i1, 'pts': pts,
                      'cum': np.r_[0.0, np.cumsum(seg)], 's': 0.0}

    def _referencia(self, s):
        """Punto y tangente unitaria del trazo en la longitud de arco s."""
        tr = self.trazo
        cum, pts = tr['cum'], tr['pts']
        if len(pts) == 1 or cum[-1] <= 0.0:
            return pts[-1].copy(), np.zeros(3)
        k = int(np.clip(np.searchsorted(cum, s, side='right') - 1, 0, len(cum) - 2))
        largo = cum[k + 1] - cum[k]
        u = 0.0 if largo <= 0 else (s - cum[k]) / largo
        xd = pts[k].copy()
        xd[:3] = pts[k, :3] + u * (pts[k + 1, :3] - pts[k, :3])
        tang = (pts[k + 1, :3] - pts[k, :3]) / largo if largo > 0 else np.zeros(3)
        return xd, tang

    def dibujar_a_velocidad(self):
        if self.trazo is None:
            if self.punto_goal >= len(self.points):
                self.phase = PHASE_FINISH
                return
            self._iniciar_trazo(self.punto_goal)
        tr, dt = self.trazo, self.dt_real

        xd, tang = self._referencia(tr['s'])
        _, x = self.compute_fk(self.q)
        error_pos = np.linalg.norm(xd[:3] - x[:3])

        # La referencia sólo avanza si el robot la va siguiendo de cerca.
        al_final = tr['s'] >= tr['cum'][-1]
        if error_pos < self.error_max_avance and not al_final:
            tr['s'] = min(tr['s'] + self.velocidad_dibujo * dt, tr['cum'][-1])
            xd, tang = self._referencia(tr['s'])
            xd_dot = np.r_[self.velocidad_dibujo * tang, 0.0, 0.0, 0.0]
        else:
            xd_dot = np.zeros(6)

        dq, x, e = dq_seguimiento(self.q, xd, xd_dot, self.K)
        self.q = self.q + np.clip(dq, -2.0, 2.0) * dt

        msg = Float64MultiArray()
        msg.data = self.q.tolist()
        self.position_pub.publish(msg)
        self.archivo.write(f"{self.get_clock().now().to_msg().sec},{x[0]},{x[1]},{x[2]}\n")

        # Fin del trazo: referencia al final y robot encima de ella.
        if tr['s'] >= tr['cum'][-1] and np.linalg.norm(e[:3]) < self.tol_fin_trazo:
            self.punto_goal = tr['i1']
            self.trazo = None
            if self.punto_goal >= len(self.points):
                self.get_logger().info("¡Llegamos al final del trazo! Pasando a fase final...")
                self.phase = PHASE_FINISH
            else:
                self.phase = PHASE_UP
                self.up_subphase = 0

    # =========================
    # IK puntual (DLS) para los movimientos del final
    # =========================
    def ik_hacia(self, q_semilla, xd, tol=1e-4, iteraciones=300):
        """q cerca de q_semilla que alcanza xd [x, y, z, quat]; None si no converge."""
        q = np.array(q_semilla, dtype=float)
        xd = np.array(xd, dtype=float)
        # Los puntos del dibujo traen el cuaternión [0, .70, .70, 0] sin
        # normalizar: normalizado, la tolerancia sí se puede alcanzar.
        xd[3:] /= np.linalg.norm(xd[3:])
        for _ in range(iteraciones):
            q = ik_dls_step(fkine_ur5, TF2xyzquat, q, xd, lamb=0.05)
            if np.linalg.norm(pose_error(xd, TF2xyzquat(fkine_ur5(q)))) < tol:
                return q
        return None

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
        # Tiempo real desde el ciclo anterior (con tope): la integración no
        # depende de cuántos ciclos por segundo alcance la PC.
        ahora = time.perf_counter()
        self.dt_real = self.dt if self._t_prev is None else min(ahora - self._t_prev, self.dt_max)
        self._t_prev = ahora

        # ── FASE 0: esperar lectura real ──────────────────────────
        #if self.phase == PHASE_INIT:
        #    return
        #print(self.phase)
        
        # Sin q_inicio no se sabe dónde está el robot: esperar a /joint_states.
        # xy y flags llegan por tópicos distintos: arrancar sólo cuando ambos
        # corresponden a la misma trayectoria (mismo número de puntos).
        if (self.points_received == True and self.temp == 1 and self.q_inicio is not None
                and len(self.flags) == len(self.points)):
            self.phase = PHASE_GOTO_START
            self.temp = 0
        
        
        
        # ── FASE 1: ir al inicio del círculo con scaled ───────────
        if self.phase == PHASE_GOTO_START:
            self.phase = PHASE_SWITCHING   # bloquear re-entrada
            
            #self.switch_my_controllers(
            #    to_activate   = 'scaled_joint_trajectory_controller',
            #    to_deactivate = 'forward_position_controller'
            #)
            
            success = self.go_to_draw_start()
            if not success:
                self.get_logger().error("No se pudo ir al inicio. Abortando.")
                return

            # ── FASE 2: cambiar a forward_position_controller ─────
            self.switch_my_controllers(
                to_activate   = 'forward_position_controller',
                to_deactivate = 'scaled_joint_trajectory_controller'
            )
            self.phase = PHASE_DRAW
            print(self.phase)
            self.get_logger().info("¡Iniciando dibujo!")
            self.punto_goal = 0
            return

        # ── FASE 3 (modo velocidad): trazo a velocidad constante ──
        if self.phase == PHASE_DRAW and self.modo_seguimiento == 'velocidad':
            self.dibujar_a_velocidad()

        # ── FASE 3: círculo con forward_position_controller ───────
        elif self.phase == PHASE_DRAW:

            # 1. Trayectoria deseada
            
            if self.punto_goal >= len(self.points):
                self.get_logger().info("¡Todos los puntos completados con éxito!")
                self.phase = PHASE_FINISH
                return
            
            xd = self.points[self.punto_goal]  
            
            # 2. Control 
            dq = compute_dq(
                q=self.q,
                xd=xd,
                K=self.K,           
            )
            
            #print(dq)
            dq_max = 2
            dq = np.clip(dq, -dq_max, dq_max)

            # 3. Integración
            self.q = self.q + dq * self.dt
            #self.q[5] = 0.0
           
            # 4. Publicar al forward controller
            msg = Float64MultiArray()
            msg.data = self.q.tolist()
            self.position_pub.publish(msg)

            # 5. FK y log
            _, x = self.compute_fk(self.q)
            #print(x)
            t = self.get_clock().now().to_msg().sec
            self.archivo.write(f"{t},{x[0]},{x[1]},{x[2]}\n")

            # 6. Marker
            #marker = set_marker_pose(self.marker, x, self)
            #if marker is not None:
            #    self.marker = marker
            #    self.marker_pub.publish(self.marker)
           
            error = np.linalg.norm(pose_error(xd, x))

            
            if np.linalg.norm(error) < self.limit_error:
                self.cont += 1
                self.punto_goal += 1
                
                
                if self.punto_goal >= len(self.points):
                    self.get_logger().info("¡Llegamos al final del trazo! Pasando a fase final...")
                    self.phase = PHASE_FINISH
                    return
            
                xd = self.points[self.punto_goal]
                flag = self.flags[self.punto_goal]
                
                if self.cont == 4:
                    self.limit_error = float(self.get_parameter('limit_error').value)
                    
                if flag == 1:
                    self.cont = 0
                    self.phase = PHASE_UP  
            
            # 8. Tiempo y debug
            self.t += self.dt
            error = np.linalg.norm(pose_error(xd, x))
            #self.get_logger().info(f"t={self.t:.2f}  error={error:.4f}")
        
        if self.phase == PHASE_UP:
            # 1. Obtener la posición cartesiana actual del robot (x)
            _, x = self.compute_fk(self.q)
    
            # -------------------------------------------------------------------------
            # PASO 1: LEVANTAR EL PLUMÓN VERTICALMENTE
            # -------------------------------------------------------------------------
            if self.up_subphase == 0:
                # Destino: Mismas coordenadas X e Y previas, pero con Z segura (0.195)
                xd = self.points[max(0, self.punto_goal - 1)].copy()
                xd[2] = self.z_arriba(xd[0], xd[1])
        
                # Ejecutar UN paso matemático de QP hacia esa posición aérea
                dq = compute_dq_qp(
                fkine=fkine_ur5, jacobian_func=jacobiano_ur5, TF2xyzquat=TF2xyzquat,
                 q=self.q, xd=xd, K=4.0, lamb=self.lamb, dt=self.dt_real
                )
                self.q += np.clip(dq, -2.0, 2.0) * self.dt_real
        
                # Evaluar si ya terminó de subir de forma reactiva
                if np.linalg.norm(pose_error(xd, x)) < 0.0008:
                    print("Paso 1 Completado: Plumón levantado.")
                    self.up_subphase = 1  # Pasamos al movimiento horizontal

    	    # -------------------------------------------------------------------------
    	    # PASO 2: MOVER HORIZONTALMENTE AL NUEVO PUNTO DE DIBUJO
     	    # -------------------------------------------------------------------------
            elif self.up_subphase == 1:
            # Destino: Nuevas coordenadas X e Y del siguiente trazo, manteniendo Z segura
                xd = self.points[self.punto_goal].copy()
                xd[2] = self.z_arriba(xd[0], xd[1])
        
                dq = compute_dq_qp(
                fkine=fkine_ur5, jacobian_func=jacobiano_ur5, TF2xyzquat=TF2xyzquat,
                q=self.q, xd=xd, K=4.0, lamb=self.lamb, dt=self.dt_real
                )
                self.q += np.clip(dq, -2.0, 2.0) * self.dt_real
        
                # Evaluar si ya llegó a la vertical del nuevo punto
                if np.linalg.norm(pose_error(xd, x)) < 0.002:
                    print("Paso 2 Completado: Robot posicionado sobre el nuevo trazo.")
                    self.up_subphase = 2  # Pasamos al descenso

           # -------------------------------------------------------------------------
    	   # PASO 3: DESCENDER AL PAPEL
    	   # -------------------------------------------------------------------------
            elif self.up_subphase == 2:
                # Destino: El punto real sobre el papel (Z original del dibujo)
                xd = self.points[self.punto_goal]
        
                dq = compute_dq_qp(
            fkine=fkine_ur5, jacobian_func=jacobiano_ur5, TF2xyzquat=TF2xyzquat,
            q=self.q, xd=xd, K=3.0, lamb=self.lamb, dt=self.dt_real
            )
                self.q += np.clip(dq, -2.0, 2.0) * self.dt_real
        
            # Evaluar si ya tocó el papel con la precisión requerida
                if np.linalg.norm(pose_error(xd, x)) < self.tol_bajada:
                    print("Paso 3 Completado: Plumón en papel. Reanudando dibujo.")
                    self.phase = PHASE_DRAW  # Regresa a la fase de dibujo continuo
                    self.up_subphase = 0     # Resetea la subfase para la próxima vez que se requiera levantar
                    return                   # Salir del ciclo actual para evitar saltos bruscos

    	# -------------------------------------------------------------------------
    	# PUBLICACIÓN CONSTANTE (Se ejecuta en cualquier subfase, 1 vez por ciclo)
    	# -------------------------------------------------------------------------
            msg = Float64MultiArray()
            msg.data = self.q.tolist()
            self.position_pub.publish(msg)
            
        if self.phase == PHASE_FINISH:
            q_fija = np.array([np.pi, -2.14, -1.34, -1.23, np.pi/2, 0.0])

            volver = self.get_parameter(
                'volver_al_inicio').get_parameter_value().bool_value
            altura = self.get_parameter(
                'altura_regreso').get_parameter_value().double_value

            self.switch_my_controllers(
                to_activate   = 'scaled_joint_trajectory_controller',
                to_deactivate = 'forward_position_controller'
            )

            # 1. Levantar en vertical sobre el último punto dibujado.
            xd_up = np.array(self.points[min(self.punto_goal, len(self.points) - 1)], dtype=float)
            xd_up[2] = self.z_papel(xd_up[0], xd_up[1]) + altura
            q_up = self.ik_hacia(self.q, xd_up)
            if q_up is not None:
                self.get_logger().info(f"Dibujo terminado. Levantando el plumón {altura*1000:.0f} mm.")
                self.send_trajectory_goal(q_up, duration_sec=1.5)
                self.q = q_up
            else:
                self.get_logger().warn("No se pudo calcular el levantamiento final; se regresa directo.")

            # 2. Regresar: sobre el punto de inicio, a la misma altura segura.
            q_start, destino = q_fija, 'postura fija'
            if volver and self.q_inicio is not None:
                _, x_ini = self.compute_fk(self.q_inicio)
                xd_ini = np.array(x_ini, dtype=float)
                xd_ini[2] = max(xd_ini[2], self.z_papel(xd_ini[0], xd_ini[1]) + altura)
                q_ini_alto = self.ik_hacia(self.q_inicio, xd_ini)
                q_start = q_ini_alto if q_ini_alto is not None else self.q_inicio
                destino = 'posición de inicio (sobre el centro de la hoja)'

            self.get_logger().info(f"Volviendo a la {destino}.")
            if self.send_trajectory_goal(q_start, duration_sec=3.0):
                self.timer.cancel()
      

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
