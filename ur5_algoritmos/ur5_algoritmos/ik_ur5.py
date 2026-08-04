#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

# Personal
from ur5_algoritmos.fk_functions import *
from ur5_algoritmos.ik_functions import *
from ur5_algoritmos.markers import *


class UR5IKNode(Node):

    def __init__(self):
        super().__init__('ur5_ik_node')

        # ===== Publisher joints =====
        self.pub = self.create_publisher(JointState, 'joint_states', 10)

        # ===== Publisher marker =====
        self.marker_pub = self.create_publisher(Marker, 'ee_marker', 10)

        # ===== Joint names (UR5) =====
        self.jnames = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        # ===== Configuración inicial =====
        self.q = np.array([np.pi/2, -1.5, -1.0, 1.2, 0.45, 0.0])

        # ===== Configuración objetiva =====
        self.targets = np.array([0.4, 0.24, 0.5, 1, 0, 0, 0])

        # ===== DLS o Pseudoinversa ===
        self.use_dls = True

        # ===== JointState =====
        self.jstate = JointState()
        self.jstate.name = self.jnames

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

        self.get_logger().info("UR5 FK Node iniciado")

    # =========================
    # FK → posición
    # =========================
    def compute_fk(self, q):
        T = fkine_ur5(q)  
        pos = TF2xyzquat(T)
        return T, pos

    # =========================
    # LOOP
    # =========================
    def update(self):

        # =========================================
        # 1. IK 
        # =========================================

        q = self.q
        xd = self.targets

        # IMPORTANTE: usar q como q0
        if self.use_dls:
            q_new = ik_dls_step(fkine_ur5, TF2xyzquat, q, xd, lamb=0.05)
        else:
            q_new = ik_pseudo_step(fkine_ur5, TF2xyzquat, q, xd, alpha=0.3)

        

        self.q = q_new

        # ===== 2. Publicar joints =====
        self.jstate.header.stamp = self.get_clock().now().to_msg()
        self.jstate.position = self.q.tolist()
        self.pub.publish(self.jstate)

        # ===== 3. FK =====
        T, pos = self.compute_fk(self.q)

        # Debug
        # print("T:\n", np.round(T, 3))

        # ===== 4. Marker =====
        self.marker = set_marker_pose(self.marker, pos, self)

        self.marker_pub.publish(self.marker)


def main(args=None):
    rclpy.init(args=args)

    node = UR5IKNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()