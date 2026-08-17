#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import WrenchStamped 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray, Bool, Float64
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from controller_manager_msgs.srv import SwitchController   # ← nuevo
import time
import os
from rclpy.executors import MultiThreadedExecutor

from ur5_algoritmos.fk_functions import *
from ur5_algoritmos.ik_functions import *
from ur5_algoritmos.kine_control_functions import *
from ur5_algoritmos.QP_functions import *
from ur5_algoritmos.markers import *


# Estados de la máquina de fases
PHASE_INIT      = 0   # esperando posición real del robot
PHASE_GOTO_START = 1  # moviendo al primer punto del círculo (scaled)
PHASE_SWITCHING  = 2  # cambiando de controller
PHASE_DOWN     = 3  # dibujando el círculo (forward)
STOP = 4
FINAL = 5

class UR5CalibrationDraw(Node):

    def __init__(self):
        super().__init__('ur5_calibration_draw')

        # ===== PARAMETROS =====
        self.dt   = 1.0 / 50.0
        self.K    = 2.5
        self.lamb = 0.01

        self.client_group = ReentrantCallbackGroup()
        self.timer_group  = MutuallyExclusiveCallbackGroup()

        # ===== Publishers / Clients =====
        self.calib_ok_pub = self.create_publisher(Float64, '/calibration_z', 10)
        
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


        # ===== Suscripción a joint_states para leer posición real =====
        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)
            
        self.force_sub = self.create_subscription(
            WrenchStamped, '/force_torque_sensor_broadcaster/wrench', self.force_cb, 10)

        # ===== Joint names =====
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # ===== Estado =====
        self.q       = None          # se llenará desde /joint_states
        self.phase   = PHASE_INIT
        self.fuerza_z = 0
        self.stop = 0
        self.x_deseado = 0
        self.t = 0
        self.archivo = open('z_calibrado.txt', 'w')
        self.calib_published = False


        # ===== Timer principal =====
        self.timer = self.create_timer(self.dt, self.update, self.timer_group)
        self.get_logger().info("Nodo iniciado. Esperando joint_states...")

    # =========================
    # Callback joint_states
    # =========================
    def joint_state_cb(self, msg: JointState):
      if self.q is not None:
        return
      try:
        self.q = np.array([
            msg.position[msg.name.index(j)]
            for j in self.joint_names
        ])
        self.get_logger().info(f"Posición real leída: {np.round(self.q, 3)}")
        self.phase = PHASE_GOTO_START
      except ValueError:
        pass
            
    
    def force_cb(self, msg: WrenchStamped):
        try:
            self.fuerza_z = msg.wrench.force.z
            #print(self.fuerza_z)
            if self.fuerza_z < -3:
                print("FUERZA ALCANZADA")
                self.stop = 1
                self.phase = STOP
        except Exception as e:
            self.get_logger().warn(f"Error en force_cb: {e}")

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
    def go_to_start(self):
        """Calcula el q inicial del círculo y mueve allí con scaled."""
        # Punto cartesiano inicial del círculo (t=0)
        self.switch_my_controllers(
                to_activate   = 'joint_trajectory_controller', 
                to_deactivate = 'forward_position_controller'
            )
            
        q_start = np.array([3.40, -2.14, -1.42, -1.13, np.pi/2, 0.0])
        
        
        # Enviar con scaled (bloqueante)
        if not self.send_trajectory_goal(q_start, duration_sec=5.0):
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
        if self.phase == PHASE_INIT:
            return

        # ── FASE 1: ir a la posicion inicial de la calibracion con scaled ───────────
        if self.phase == PHASE_GOTO_START:
            self.phase = PHASE_SWITCHING   # bloquear re-entrada
            success = self.go_to_start()
            if not success:
                self.get_logger().error("No se pudo ir al inicio. Abortando.")
                return

            # ── FASE 2: cambiar a forward_position_controller ─────
            self.switch_my_controllers(
                to_activate   = 'forward_position_controller',
                to_deactivate = 'joint_trajectory_controller'
            )
            
            #Posicion Inicial efector final
            _, x_initial = self.compute_fk(self.q)
            
            self.x_deseado = x_initial.copy()
            self.x_deseado[2] = x_initial[2] - 0.005
            self.x_deseado[3] = 0.0
            self.x_deseado[6] = 0.0
            print(self.x_deseado)
            
            self.phase = PHASE_DOWN
            #print(self.phase)
            self.get_logger().info("¡Iniciando calibracion!")
            return

        # ── FASE 3: Inicio de calibración ───────
        if self.phase == PHASE_DOWN and self.stop == 0:

            # 1. Trayectoria deseada
            # x_deseado

            # 2. Control 
            dq = compute_dq(
                q=self.q,
                xd=self.x_deseado,
                K=self.K,           
            )
            
            #print(dq)
            dq_max = 0.35
            dq = np.clip(dq, -dq_max, dq_max)

            # 3. Integración
            self.q = self.q + dq * self.dt
            #self.q[0] = 0.0
            #print(self.q)

            # 4. Publicar al forward controller
            msg = Float64MultiArray()
            msg.data = self.q.tolist()
            self.position_pub.publish(msg)

            # 5. FK y log
            _, x = self.compute_fk(self.q)
            print(x)
            
            t = self.get_clock().now().to_msg().sec
            
            error = np.linalg.norm(pose_error(self.x_deseado, x))
            print("bajando")
            # 7. Evaluar error y salto de dibujo
            if np.linalg.norm(error) < 0.001:
                print(self.fuerza_z)
                self.x_deseado[2] = self.x_deseado[2] - 0.007
                 
                    
            # 7. Tiempo y debug
            self.t += self.dt
            #self.get_logger().info(f"t={self.t:.2f}  error={error:.4f}")
        
        if self.phase == STOP and self.stop == 1:
            _, x_final = self.compute_fk(self.q)
            self.get_logger().info(f"CALIBRACIÓN COMPLETADA — q={np.round(self.q,3)}")
            z_draw = x_final[2]
            self.get_logger().info(f"CALIBRACIÓN COMPLETADA — Z dibujo={z_draw:.4f} m")
            msg = Float64()
            msg.data = z_draw
            self.calib_ok_pub.publish(msg)
            
            self.archivo.write(str(z_draw))
           
            # Ahora sí cierras el nodo con seguridad
            self.get_logger().info(f"Calibración guardada ({z_draw} m). Cerrando...")
            
            self.switch_my_controllers(
                to_activate   = 'joint_trajectory_controller', 
                to_deactivate = 'forward_position_controller'
            )
            
            q_start = np.array([np.pi, -2.14, -1.34, -1.23, np.pi/2, 0.0])
            # Enviar con scaled (bloqueante)
            self.send_trajectory_goal(q_start, duration_sec=3.0)
            self.calib_published = True
            self.phase = FINAL
        
        if self.phase == FINAL:
            
            self.get_logger().info("INICIANDO DIBUJO")
            self.timer.cancel()
            rclpy.shutdown()
            
        
                    


def main(args=None):
    rclpy.init(args=args)
    node = UR5CalibrationDraw()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        #node.archivo.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        



if __name__ == '__main__':
    main()
