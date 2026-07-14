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
from ur5_algoritmos.fk_functions import *
from ur5_algoritmos.ik_functions import *
from ur5_algoritmos.kine_control_functions import *

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
        
        self.action_client = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory', callback_group=self.client_group)
        
        self.switch_ctrl_client = self.create_client(SwitchController, '/controller_manager/switch_controller', callback_group=self.client_group)

        self.get_logger().info("Nodo de paneo ROS 2 iniciado.")
        
        # Variable de inicio ----
        self.error_drink = 100
        self.state = "SCANNING_A"
        self.K    = 1
        self.temp = 0
        self.step = 0
        
        #Implementacion
        #self.R_eff_cam = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.R_eff_cam = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # Tiempo de paneo por lado
        self.T = 10 
        
        
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
        
    def compute_fk(self, q):
        T = fkine_ur5(q)
        x = TF2xyzquat(T)
        return T, x
          

    def perform_scanning(self, p0, pf, quat_deseada):
        
        # Calcular trayectoria deseada
        s, sdot = self.s_and_sdot(self.step, self.T)
        
        xd = p0 + s * (pf - p0)
        xd = np.concatenate((xd, quat_deseada))
        
        xd_dot = sdot * (pf - p0)
        xd_dot = np.concatenate((xd_dot, np.zeros(3)))
        
        
        dq = compute_dq_trayectoria(
                q=self.q,
                xd=xd,
                xd_dot=xd_dot,
                K=self.K,           
            )
        dq = np.clip(dq, -0.5, 0.5)

              
        self.q = self.q + dq*self.dt
        self.q[5] = 0.0
        #print(self.q)
        
        DH06, x = self.compute_fk(self.q)
        x_pos = x[0:3]
        R = DH06[0:3, 0:3]
        print(x)
            

        # Movimiento UR5
        #self.send_trajectory_goal(q, delay)
        msg = Float64MultiArray()
        msg.data = self.q.tolist()
        self.position_pub.publish(msg)


        # Captura de QRs
        if self.pos_qr_raw is not None and len(self.pos_qr_raw) >= 4:
        
            datos_locales = list(self.pos_qr_raw)
            num_qrs = len(datos_locales) // 4
            for j in range(num_qrs):
                idx = j * 4
                if idx + 3 >= len(datos_locales):
                    break
                pos_cam = np.array([datos_locales[idx], datos_locales[idx+1], datos_locales[idx+2]])
                qr_id = int(datos_locales[idx+3])
                    
                # Transformación a la base del robot
                
                pos_base_qr = (R @ self.R_eff_cam @ pos_cam) + x_pos
                pos_base_qr = np.concatenate([pos_base_qr, x[3:7]])
                
                self.qr_global_positions[qr_id] = pos_base_qr.copy()
                self.get_logger().info(f"QR {qr_id} guardado en paso {self.step}")
            
        
        print("Posiciones guardadas:", self.qr_global_positions)
        
        
    def serve_drink(self, xd):
        
        dq = compute_dq(
                q=self.q,
                xd=xd,
                K=self.K,           
            )
            
        dq = np.clip(dq, -0.5, 0.5)
        self.q = self.q + dq*self.dt
        
        msg = Float64MultiArray()
        msg.data = self.q.tolist()
        self.position_pub.publish(msg)
        
        DH06, x = self.compute_fk(self.q)
        
        error = np.linalg.norm(pose_error(xd, x))
        
        return error
        
        


    def main_loop(self):
        if self.is_busy:
            return
        
        self.is_busy = True
        
        if self.state == "SCANNING_A":
    
            if self.temp == 0:
                #Posicion inicial
                self.switch_my_controllers('joint_trajectory_controller','forward_position_controller')
                
                #self.q = np.array([-0.49, -1.55, -2.28, -2.34, -0.5, 0.0])
                self.q = np.array([1.57, -1.22, 1.9, -0.63, 1.50, 0])
                self.send_trajectory_goal(self.q, 5.0)
                print("cambiando_Controlador")
                self.switch_my_controllers('forward_position_controller', 'joint_trajectory_controller')
                print("posicion_inicial")
                
        
        
            if self.step < self.total_step:
                print("scanA")
                self.temp = 1
                
                p0 = np.array([0.10, -0.53, 0.18])
                pf = np.array([-0.45, -0.53, 0.18])
        
                #R_deseada = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                quat_deseado = np.array([0.707, 0.707, 0, 0])
                
                self.perform_scanning(p0, pf, quat_deseado)
                self.step = self.step + 1
            else:
                self.get_logger().info("Escaneo A completado")
                self.state = "SCANNING_B"
                self.step = 0
                self.temp = 0
             
        if self.state == "SCANNING_B":
             if self.temp == 0:
                 self.switch_my_controllers('joint_trajectory_controller','forward_position_controller')
                 self.q = np.array([0.563, -1.16, 1.63, -0.52, 2.08, 0.0])
                 self.send_trajectory_goal(self.q,2)
                
                 self.switch_my_controllers('forward_position_controller', 'joint_trajectory_controller')
         
        
             if self.step < self.total_step:
                 print("scanB")
                 self.temp = 1
                 
                 p0 = np.array([-0.42, -0.32, 0.20])
                 pf = np.array([-0.42, 0.32, 0.20])
        
                 #R_deseada = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
                 
                 quat_deseado = np.array([0.5, 0.5, -0.5, -0.5])
                 
                 self.perform_scanning(p0, pf, quat_deseado)
                 self.step = self.step + 1
             else:
                 self.state = "SCANNING_C"
                 self.temp = 0
                 self.step = 0
                 
        if self.state == "SCANNING_C":
             if self.temp == 0:
                 self.switch_my_controllers('joint_trajectory_controller','forward_position_controller')
                 self.q = np.array([-0.99, -1.29,  1.84,  -0.55,  1.2,  0])
                 self.send_trajectory_goal(self.q,2)
                
                 self.switch_my_controllers('forward_position_controller', 'joint_trajectory_controller')
         
        
             if self.step < self.total_step:
                 print("scanC")
                 self.temp = 1
                 
                 p0 = np.array([-0.42, 0.32, 0.20])
                 pf = np.array([0.10, 0.32, 0.20])
        
                 #R_deseada = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
                 
                 quat_deseado = np.array([0.0, 0.0, 0.707, 0.707])
                 
                 self.perform_scanning(p0, pf, quat_deseado)
                 self.step = self.step + 1
             else:
                 self.state = "GUI"
                 self.temp = 0
                 self.step = 0
                              
        elif self.state == "GUI":
            print("---------SELECCION DE BEBIDA------------")
            print(self.qr_global_positions)
            self.valor = int(input("Ingrese valor: "))
            if self.valor == 100 or self.valor == 200 or self.valor == 300:
                if self.qr_global_positions[self.valor] is None:
                    self.get_logger().error("ID no encontrado.")
                    return

                print("Eleccion correcta")   
                self.state = "SERVING"
            
            
        elif self.state == "SERVING":
            print("sirviendo")
            x_des = self.qr_global_positions[self.valor].copy()
            # Offset de aproximación
            #x_des[0] += 0.15
            
            if self.error_drink > 0.001:
                self.error_drink = self.serve_drink(x_des)
                self.state = "SERVING"
            else:
                self.error_drink = 100
                self.state = "GUI"
        
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
