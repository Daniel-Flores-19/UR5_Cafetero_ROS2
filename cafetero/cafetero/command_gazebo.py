#!/usr/bin/env python3
import rclpy
from simple_actions import SimpleActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


def main():
    # Iniciar un nodo
    rclpy.init()
    node = rclpy.create_node('command_simple')

    # Declara la acción del tipo cliente
    client = SimpleActionClient(node, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')
    
    node.get_logger().info(f"Conectando...")
    client.wait_for_server()
    node.get_logger().info(f"Enviando...")
    
    # Declara las variables del brazo robótico
    # Lista de nombres
    joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    # Lista de valores de la configuración del robot
    Q0 = [0.0, -0.785, 0.785, -1.57, -1.57, 0.0]

    # Definir el tipo de mensaje a utilizar
    goal = FollowJointTrajectory.Goal()
    traj = JointTrajectory()
    point = JointTrajectoryPoint()

    # Definir los nombres de las articulaciones de traj.joint_names
    traj.joint_names = joint_names

    # Definir la posición inicial de point
    point.positions = Q0
    point.velocities = [0.0] * 6
    point.time_from_start = Duration(sec=3)

    # Agregar el punto a la trayectoria
    traj.points.append(point)
    goal.trajectory = traj

    # Enviar el objetivo
    client.send_goal(goal)
    
    
    # Imprimir el resultado
    node.get_logger().info(f"Completado")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
