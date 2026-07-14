#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import numpy as np
import time

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float32MultiArray
from controller_manager_msgs.srv import SwitchController
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from functions import fkine_ur5, jacobian_ur5

class UR5ScannerNode(Node):
    def __init__(self):
        super().__init__('ur5_scanner_node')
        
        # --- Variables de estado ---
        self.pos_qr_raw = None
        self.num_qr = 0
        self.qr_global_positions = {100: None, 200: None, 300: None, 400: None}
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        self.client_group = ReentrantCallbackGroup()
        
        self.timer_group = MutuallyExclusiveCallbackGroup()
        
        # --- Publicador de posicion
        self.position_pub = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        
        # --- Suscriptores y Clientes ---
        self.subscription = self.create_subscription(Float32MultiArray, "/qr_pose", self.qr_pose_callback, 10, callback_group=self.client_group)
        
        self.action_client = ActionClient(self, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory', callback_group=self.client_group)
        
        self.switch_ctrl_client = self.create_client(SwitchController, '/controller_manager/switch_controller', callback_group=self.client_group)

        self.get_logger().info("Nodo de paneo ROS 2 iniciado.")
        
        # Variable de inicio ----
        
        self.state = "SCANNING_A"
        self.temp = 0
        self.step = 0
        self.R_eff_cam = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        
        # Tiempo de paneo por lado
        self.T = 8 
        
        
        # Frecuencia de lazo
        self.f = 25   # Hz
        self.dt = 1.0 / self.f
        
        #Total de pasos
        self.total_step = int(self.T * self.f)
       
        self.is_busy = False
        
        
        self.timer = self.create_timer(self.dt, self.main_loop, callback_group=self.timer_group)
        
        
    def switch_my_controllers(self, to_activate, to_deactivate):
        while not self.switch_ctrl_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando al servicio SwitchController...')

        request = SwitchController.Request()
        request.activate_controllers = [to_activate]
        request.deactivate_controllers = [to_deactivate]
        request.strictness = SwitchController.Request.STRICT

        self.get_logger().info(f"Cambiando: Activar {to_activate}, Desactivar {to_deactivate}")
        future = self.switch_ctrl_client.call_async(request)
        
        # Espera manual del future compatible con MultiThreadedExecutor
        while rclpy.ok():
            if future.done():
                self.get_logger().info("Controladores cambiados con éxito.")
                return future.result()
            time.sleep(0.1)
        


    def qr_pose_callback(self, msg):
        self.pos_qr_raw = msg.data
        self.num_qr = len(msg.data) // 4 if len(msg.data) >= 4 else 0
    
    def s_and_sdot(self, t, T):
        tau = np.clip(t / T, 0.0, 1.0)
        s = 3*tau**2 - 2*tau**3
        ds_dt = (6*tau*(1-tau)) / T
        return s, ds_dt
    
        
    def send_trajectory_goal(self, q_target, duration_sec):
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Servidor de acción no disponible.")
            return False

        goal_msg = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = q_target.tolist()
        point.velocities = [0.0] * 6
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)

        trajectory.points.append(point)
        goal_msg.trajectory = trajectory

        self.get_logger().info(f"Enviando objetivo de trayectoria...")
        send_goal_future = self.action_client.send_goal_async(goal_msg)

        # 1. Esperar a que el servidor acepte el Goal
        while rclpy.ok():
            print("hola")
            if send_goal_future.done():
                break
            time.sleep(0.1)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Objetivo RECHAZADO")
            return False

        # 2. Esperar a que el robot termine de moverse
        self.get_logger().info("Objetivo aceptado, moviendo robot...")
        result_future = goal_handle.get_result_async()
        
        while rclpy.ok():
            if result_future.done():
                self.get_logger().info("¡Movimiento finalizado!")
                return True
            time.sleep(0.1)
        
        

    def perform_scanning(self, p0, pf, R_deseada):
        
        # Calcular trayectoria deseada
        s, sdot = self.s_and_sdot(self.step, self.T)
        xd = p0 + s * (pf - p0)
        xd_dot = sdot * (pf - p0)

        # Cinemática directa actual
        DH06 = fkine_ur5(self.q)
        x = DH06[0:3, 3]
        print(x)
        R = DH06[0:3, 0:3]

        # Cálculo de errores
        e_pos = xd - x
        R_err = R_deseada @ R.T
        e_rot = 0.5 * np.array([
                R_err[2,1] - R_err[1,2],
                R_err[0,2] - R_err[2,0],
                R_err[1,0] - R_err[0,1]
            ])

        # Control cinemático
        v = np.concatenate([xd_dot + 1.0 * e_pos, 1.0 * e_rot])
        Jac = jacobian_ur5(self.q)
        dq = np.linalg.pinv(Jac) @ v
        dq = np.clip(dq, -0.5, 0.5)      
        self.q = self.q + dq*self.dt
        print(self.q)
            

        # Movimiento UR5
        #self.send_trajectory_goal(q, delay)
        msg = Float64MultiArray()
        msg.data = self.q.tolist()
        self.position_pub.publish(msg)


        # Captura de QRs
        if self.pos_qr_raw is not None and len(self.pos_qr_raw) >= 4:
            num_qrs = len(self.pos_qr_raw) // 4
            for j in range(num_qrs):
                idx = j * 4
                pos_cam = np.array([self.pos_qr_raw[idx], self.pos_qr_raw[idx+1], self.pos_qr_raw[idx+2]])
                qr_id = int(self.pos_qr_raw[idx+3])
                    
                # Transformación a la base del robot
                
                pos_base_qr = (R @ self.R_eff_cam @ pos_cam) + x
                self.qr_global_positions[qr_id] = pos_base_qr.copy()
                self.get_logger().info(f"QR {qr_id} guardado en paso {self.step}")
            
        
        print("Posiciones guardadas:", self.qr_global_positions)
        
        
    def serve_drink(self, qr_id):
        """Control cinematico hacia los bidones"""
        if self.qr_global_positions[qr_id] is None:
            self.get_logger().error("ID no encontrado.")
            return

        x_des = self.qr_global_positions[qr_id].copy()
        x_des[1] += 0.20 # Offset de aproximación
        
        r_deseada = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        dt = 0.1
        q = np.array([-0.49, -1.55, -2.28, -2.34, -0.5, 0.0]) # Pos inicial o actual
        
        error = 1.0
        while error > 0.01:
            DH06 = fkine_ur5(q)
            x = DH06[0:3, 3]
            r = DH06[0:3, 0:3]

            e_pos = x_des - x
            r_err = r_deseada @ r.T
            e_rot = 0.5 * np.array([r_err[2,1]-r_err[1,2], r_err[0,2]-r_err[2,0], r_err[1,0]-r_err[0,1]])

            v = np.concatenate([0.3 * e_pos, 0.3 * e_rot])
            jac = jacobian_ur5(q)
            dq = np.linalg.pinv(jac) @ v
            q = q + dt * dq
            
            error = np.linalg.norm(e_pos)
            self.send_trajectory_goal(q, 0.5)
            self.get_logger().info(f"Error actual: {error:.4f}")

    def main_loop(self):
        if self.is_busy:
            return
        
        self.is_busy = True
        
        if self.state == "SCANNING_A":
    
            if self.temp == 0:
                self.q = np.array([-0.49, -1.55, -2.28, -2.34, -0.5, 0.0])
                self.send_trajectory_goal(self.q, 1.0)
                print("cambiando_Controlador")
                self.switch_my_controllers('forward_position_controller', 'scaled_joint_trajectory_controller')
                print("posicion_inicial")
                
        
        
            if self.step < self.total_step:
                print("scanA")
                self.temp = 1
                
                p0 = np.array([0.22, -0.32, 0.18])
                pf = np.array([-0.22, -0.32, 0.18])
        
                R_deseada = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                
                
                self.perform_scanning(p0, pf, R_deseada)
                self.step = self.step + 1
            else:
                self.get_logger().info("Escaneo A completado")
                self.state = "SCANNING_B"
                self.step = 0
                self.temp = 0
             
        if self.state == "SCANNING_B":
             if self.temp == 0:
                 self.switch_my_controllers('scaled_joint_trajectory_controller','forward_position_controller')
                 self.q = np.array([-2.06, -1.91, -1.74, -2.62, -0.52, 0.0])
                 self.send_trajectory_goal(self.q,2)
                
                 self.switch_my_controllers('forward_position_controller', 'scaled_joint_trajectory_controller')
         
        
             if self.step < self.total_step:
                 print("scanB")
                 self.temp = 1
                 
                 p0 = np.array([-0.42, -0.32, 0.20])
                 pf = np.array([-0.42, -0.20, 0.20])
        
                 R_deseada = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
                
                 
                 self.perform_scanning(p0, pf, R_deseada)
                 self.step = self.step + 1
             else:
                 self.state = "IDLE"
                 self.temp = 0
                 self.step = 0
                     
        elif self.state == "IDLE":
            pass
        elif self.state == "SERVING":
            pass
        
        self.is_busy = False


def main(args=None):
    rclpy.init(args=args)

    node = UR5ScannerNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
