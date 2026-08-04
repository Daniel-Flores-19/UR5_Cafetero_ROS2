#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState

# FK
from ur5_algoritmos.fk_functions import *
from ur5_algoritmos.markers import *


class UR5FKNode(Node):

    def __init__(self):
        super().__init__('ur5_fk_node')

        # ===== Estado real =====
        self.q = None
        self.q_initialized = False
        self.command_sent = False   # 🔥 clave

        # ===== Publisher =====
        self.pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # ===== Subscriber =====
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        # ===== Marker =====
        self.marker_pub = self.create_publisher(Marker, 'ee_marker', 10)

        # ===== Joint names =====
        self.jnames = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        # ===== Configuración deseada =====
        self.q_target = np.array([3.8, -1.8, -2.0, -0.6, 1.0, 0.0])

        # ===== Marker =====
        self.marker = create_sphere_marker(
            frame="base_link",
            ns="end_effectors",
            marker_id=1,
            scale=0.05,
            color=(0.0, 0.0, 1.0, 1.0)
        )

        # ===== Timer =====
        self.timer = self.create_timer(1.0 / 20.0, self.update)

        self.get_logger().info("UR5 FK Node (Gazebo) iniciado")

    # =========================
    def joint_callback(self, msg):
        if not self.q_initialized:
            self.q = np.array(msg.position)
            self.q_initialized = True
            self.get_logger().info("Estado sincronizado con Gazebo")

    # =========================
    def compute_fk(self, q):
        T = fkine_ur5(q)
        pos = TF2xyzquat(T)
        return T, pos

    # =========================
    def send_trajectory_once(self):

        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.jnames

        point = JointTrajectoryPoint()
        point.positions = self.q_target.tolist()

        # 🔥 TIEMPO FIJO (CLAVE)
        point.time_from_start.sec = 2
        point.time_from_start.nanosec = 0

        traj_msg.points.append(point)
        self.pub.publish(traj_msg)

        self.get_logger().info("Trayectoria enviada")

    # =========================
    def update(self):

        if not self.q_initialized:
            return

        # 🔥 SOLO enviar UNA VEZ
        if not self.command_sent:
            self.send_trajectory_once()
            self.command_sent = True

        # ===== Usar estado real del robot =====
        T, pos = self.compute_fk(self.q)

        print("T:\n", np.round(T, 3))

        # ===== Marker =====
        self.marker = set_marker_pose(self.marker, pos, self)
        self.marker_pub.publish(self.marker)


def main(args=None):
    rclpy.init(args=args)

    node = UR5FKNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()