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
from std_msgs.msg import Float64MultiArray, Bool, Float64
from visualization_msgs.msg import Marker
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from controller_manager_msgs.srv import SwitchController  
import time
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32MultiArray, Int8MultiArray, MultiArrayDimension

from ur5_algoritmos.fk_functions import *
from ur5_algoritmos.ik_functions import *
from ur5_algoritmos.kine_control_functions import *
from ur5_algoritmos.QP_functions import *
from ur5_algoritmos.markers import *


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
            '/joint_trajectory_controller/follow_joint_trajectory',
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
        
        # Leer el parámetro que le envió el archivo Launch
        self.dist_z = self.get_parameter('dist_z').get_parameter_value().double_value
        
        self.get_logger().info(f"Altura Z recibida desde calibración: {self.dist_z}")
        
        
        # ===== Estado =====
        self.q       = None          
        self.phase   = PHASE_INIT

        # ===== Marker =====
        self.marker = create_sphere_marker(
            frame="base_link_inertia", ns="ee", marker_id=0,
            scale=0.05, color=(0.0, 0.0, 1.0, 1.0))

        self.archivo = open('posiciones_robot.txt', 'w')
        self.archivo.write('timestamp,x,y,z\n')

        self.t  = 0.0
        self.temp = 1
        
        # Distancia del efector a la mesa -------------
        self.dist_z = self.dist_z + 0.0016
        
        self.dist_z_up = self.dist_z * 1.06
        
        
        self.cont = 0
        self.limit_error = 0.002
        self.up_subphase = 0
        self. flag_calibration = 0
        
        
        self.points_received = False

        # ===== Timer principal =====
        self.timer = self.create_timer(self.dt, self.update, self.timer_group)
        self.get_logger().info("Nodo iniciado. Esperando joint_states y puntos de dibujo ...")

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
            self.get_logger().info(f"Posición real leída: {np.round(self.q, 3)}")
            self.phase = 6 #PHASE_GOTO_START   # arrancar fase 1
        except ValueError:
            pass
           
        
        
    def xy_callback(self, msg):

        data = msg.data
        self.points = []

        for i in range(0, len(data), 2):

            x = data[i]
            y = data[i+1]
            z = self.dist_z
            self.points.append([x, y, z, 0, 0.70 , 0.70, 0])

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
    # Mover al inicio del círculo (bloqueante, solo una vez)
    # =========================
    def go_to_draw_start(self):
        """Calcula el q inicial el dibujo y mueve allí con scaled."""
        # Punto cartesiano inicial del círculo (t=0)
        #xd_start = circular_trajectory(0.0, self.center, self.radius, self.omega)
        
        
        #Posicion Inicial para DIBUJAR
        q_start = np.array([np.pi, -2.14, -1.34, -1.23, np.pi/2, 0.0])
        self.send_trajectory_goal(q_start, duration_sec = 5.0)

        xd_start = self.points[0]
        
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
                break
        
        
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
        #if self.phase == PHASE_INIT:
        #    return
        #print(self.phase)
        
        if self.points_received == True and self.temp == 1:
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
                to_deactivate = 'joint_trajectory_controller'
            )
            self.phase = PHASE_DRAW
            print(self.phase)
            self.get_logger().info("¡Iniciando dibujo!")
            self.punto_goal = 0
            return

        # ── FASE 3: círculo con forward_position_controller ───────
        if self.phase == PHASE_DRAW:

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
                    self.limit_error=0.002
                    
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
                xd[2] = self.dist_z_up
        
                # Ejecutar UN paso matemático de QP hacia esa posición aérea
                dq = compute_dq_qp(
                fkine=fkine_ur5, jacobian_func=numerical_jacobian, TF2xyzquat=TF2xyzquat,
                 q=self.q, xd=xd, K=4.0, lamb=self.lamb, dt=self.dt
                )
                self.q += np.clip(dq, -2.0, 2.0) * self.dt
        
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
                xd[2] = self.dist_z_up
        
                dq = compute_dq_qp(
                fkine=fkine_ur5, jacobian_func=numerical_jacobian, TF2xyzquat=TF2xyzquat,
                q=self.q, xd=xd, K=4.0, lamb=self.lamb, dt=self.dt
                )
                self.q += np.clip(dq, -2.0, 2.0) * self.dt
        
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
            fkine=fkine_ur5, jacobian_func=numerical_jacobian, TF2xyzquat=TF2xyzquat,
            q=self.q, xd=xd, K=3.0, lamb=self.lamb, dt=self.dt
            )
                self.q += np.clip(dq, -2.0, 2.0) * self.dt
        
            # Evaluar si ya tocó el papel con la precisión requerida
                if np.linalg.norm(pose_error(xd, x)) < 0.0001:
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
            q_start = np.array([np.pi, -2.14, -1.34, -1.23, np.pi/2, 0.0])
            print("Fase Final")
            self.switch_my_controllers(
                to_activate   = 'joint_trajectory_controller',
                to_deactivate = 'forward_position_controller'
            )
            
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
