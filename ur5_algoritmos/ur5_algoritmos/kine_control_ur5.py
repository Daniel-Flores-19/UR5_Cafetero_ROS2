#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

# Personal
from ur5_algoritmos.fk_functions import *
from ur5_algoritmos.ik_functions import *
from ur5_algoritmos.kine_control_functions import *
from ur5_algoritmos.markers import *


class UR5ControlNode(Node):

    def __init__(self):
        super().__init__('ur5_kinecontrol_node')

        # ===== PARAMETROS =====
        self.dt = 1.0 / 50.0   # 50 Hz
        self.K = 1.5           # ganancia

        # ===== Publisher joints =====
        self.pub = self.create_publisher(JointState, 'joint_states', 10)

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

        # ===== Estado inicial =====
        self.q = np.array([np.pi/2, -1.5, -1.0, 1.2, 0.45, 0.0])

        # ===== Objetivo =====
        self.xd = np.array([0.4, 0.24, 0.5, 1, 0, 0, 0])

        # ===== JointState =====
        self.jstate = JointState()
        self.jstate.name = self.jnames

        # ===== Marker =====
        self.marker = create_sphere_marker(
            frame="base_link",
            ns="ee",
            marker_id=0,
            scale=0.05,
            color=(0.0, 0.0, 1.0, 1.0)
        )

        # ===== Trayectoria circular =====
        self.t = 0.0
        self.radius = 0.1
        self.omega = 1.0  # rad/s

        # centro de la circunferencia
        self.center = np.array([0.4, 0.24, 0.5])

        # ===== Timer =====
        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info("UR5 Control Node iniciado")

    # =========================
    # FK
    # =========================
    def compute_fk(self, q):
        T = fkine_ur5(q)
        x = TF2xyzquat(T)
        return T, x

    # =========================
    # LOOP
    # =========================
    def update(self):

        # ===== Trayectoria =====
        xd = circular_trajectory(self.t, self.center, self.radius, self.omega)

        # ===== Control =====
        dq = compute_dq(self.q, xd, self.K)

        # ===== Integración =====
        self.q = self.q + dq * self.dt

        # ===== Publicar joints =====
        self.jstate.header.stamp = self.get_clock().now().to_msg()
        self.jstate.position = self.q.tolist()
        self.pub.publish(self.jstate)

        # ===== FK =====
        T, x = self.compute_fk(self.q)

        # ===== Marker =====
        marker = set_marker_pose(self.marker, x, self)

        if marker is not None:
            self.marker = marker
            self.marker_pub.publish(self.marker)

        # ===== Tiempo =====
        self.t += self.dt

        # Debug
        print("Error:", np.linalg.norm(pose_error(xd, x)))


def main(args=None):
    rclpy.init(args=args)

    node = UR5ControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()