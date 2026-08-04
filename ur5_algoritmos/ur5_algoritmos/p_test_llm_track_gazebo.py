#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState

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
        self.dt = 1.0 / 20.0
        self.K = 0.5
        self.time_acc = 0.0

        # ===== Trayectoria =====
        self.traj = traj
        self.index = 0

        # ===== Estado real =====
        self.q = None
        self.q_initialized = False

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

        # ===== Markers =====
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

        self.traj_marker.header.stamp = self.get_clock().now().to_msg()
        self.traj_marker_pub.publish(self.traj_marker)

        # ===== Timer =====
        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info(f"UR5 Control Node iniciado con {len(self.traj)} puntos")

    # =========================
    def joint_callback(self, msg):
        if not self.q_initialized:
            self.q = np.array(msg.position)
            self.q_initialized = True
            self.get_logger().info("Estado inicial sincronizado con Gazebo")

            # ===== Alinear trayectoria =====
            _, x = self.compute_fk(self.q)

            # reemplazar primer punto
            self.traj[0] = x

            # buscar punto más cercano
            distances = [np.linalg.norm(pose_error(p, x)) for p in self.traj]
            self.index = int(np.argmin(distances))

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
    def compute_fk(self, q):
        T = fkine_ur5(q)
        x = TF2xyzquat(T)
        return T, x

    # =========================
    def update(self):

        if not self.q_initialized:
            return

        # ===== Trayectoria =====
        xd = self.traj[self.index]

        # ===== FK =====
        T, x = self.compute_fk(self.q)

        # ===== Error =====
        err = np.linalg.norm(pose_error(xd, x))

        # ===== Avance adaptativo =====
        if err < 0.03:
            self.index += 2      # muy cerca → acelera
        elif err < 0.08:
            self.index += 1      # cerca → normal
        # si está lejos → NO avances

        self.index = self.index % len(self.traj)
        
        # ===== Control =====
        try:
            dq = compute_dq(self.q, xd, self.K)
            dq = np.clip(dq, -0.5, 0.5)
        except Exception as e:
            self.get_logger().warn(f"IK error: {e}")
            dq = np.zeros(6)

        # ===== Integración =====
        self.q = self.q + dq * self.dt

        # ===== Tiempo =====
        self.time_acc += self.dt

        # ===== Trajectory msg =====
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.jnames

        point = JointTrajectoryPoint()
        point.positions = self.q.tolist()

        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(self.dt * 1e9)

        traj_msg.points.append(point)

        self.pub.publish(traj_msg)

        # ===== Marker =====
        marker = set_marker_pose(self.marker, x, self)

        if marker is not None:
            self.marker = marker
            self.marker_pub.publish(self.marker)

        print("Error:", err)


# =========================
def main(args=None):

    user_input = input("Ingresa figura: ")
    traj3d = get_trajectory_3d(user_input)

    print("Trayectoria generada:", len(traj3d), "puntos")

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