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

# LLM
from ur5_algoritmos.p_llm_interface import get_trajectory_3d


class UR5ControlNode(Node):

    def __init__(self, traj):
        super().__init__('ur5_kinecontrol_node')

        # ===== PARAMETROS =====
        self.dt = 1.0 / 50.0
        self.K = 2

        # ===== Trayectoria =====
        self.traj = traj
        self.index = 0

        # ===== Publishers =====
        self.pub = self.create_publisher(JointState, 'joint_states', 10)
        self.marker_pub = self.create_publisher(Marker, 'ee_marker', 10)
        self.traj_marker_pub = self.create_publisher(Marker, 'traj_marker', 10)

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

        # ===== JointState =====
        self.jstate = JointState()
        self.jstate.name = self.jnames

        # ===== Marker EE =====
        self.marker = create_sphere_marker(
            frame="base_link",
            ns="ee",
            marker_id=0,
            scale=0.05,
            color=(0.0, 0.0, 1.0, 1.0)
        )

        # ===== Marker Trayectoria =====
        self.traj_marker = create_line_marker(frame="base_link")
        self.traj_marker = self.load_trajectory(self.traj_marker, self.traj)

        # Publicar una vez
        self.traj_marker.header.stamp = self.get_clock().now().to_msg()
        self.traj_marker_pub.publish(self.traj_marker)

        # ===== Timer =====
        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info(f"UR5 Control Node iniciado con {len(self.traj)} puntos")

    # =========================
    # Cargar trayectoria al marker
    # =========================
    def load_trajectory(self, marker, traj3d):

        from geometry_msgs.msg import Point

        marker.points.clear()

        for pose in traj3d:
            p = Point()
            p.x = float(pose[0])
            p.y = float(pose[1])
            p.z = float(pose[2])
            marker.points.append(p)

        return marker

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
        xd = self.traj[self.index]

        # ===== FK (ANTES del error) =====
        T, x = self.compute_fk(self.q)

        # ===== Error =====
        err = np.linalg.norm(pose_error(xd, x))

        #if err < 0.02:
        self.index += 1   # saltas puntos

        # loop circular (opcional)
        self.index = self.index % len(self.traj)

        # ===== Control =====
        try:
            dq = compute_dq(self.q, xd, self.K)
        except:
            dq = np.zeros(6)

        # ===== Integración =====
        self.q = self.q + dq * self.dt

        # ===== Publicar joints =====
        self.jstate.header.stamp = self.get_clock().now().to_msg()
        self.jstate.position = self.q.tolist()
        self.pub.publish(self.jstate)

        # ===== Marker =====
        marker = set_marker_pose(self.marker, x, self)

        if marker is not None:
            self.marker = marker
            self.marker_pub.publish(self.marker)

        # ===== Debug =====
        print("Error:", err)


# =========================
# MAIN
# =========================
def main(args=None):

    # ===== LLM =====
    user_input = input("Ingresa figura: ")
    traj3d = get_trajectory_3d(user_input)

    print("Trayectoria generada:", len(traj3d), "puntos")

    # ===== ROS =====
    rclpy.init(args=args)

    node = UR5ControlNode(traj3d)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()