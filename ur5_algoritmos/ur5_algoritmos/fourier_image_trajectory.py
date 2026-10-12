#!/usr/bin/env python3
"""
fourier_image_trajectory.py — dibuja una imagen (line art) con trazos Fourier.

Pipeline en trayectorias/fourier_image_functions.py: líneas por su eje
(esqueleto refinado), rellenos negros con contornos concéntricos, sombreado en
zigzag de zonas oscuras y bordes de color sin línea; cada camino es una serie
de Fourier con waypoints cada paso_waypoint_mm.

Como painting_trajectory, el centro del dibujo es el XY donde está el efector
al arrancar (primer /joint_states + FK), y publica en letter_trajectory/xy y
letter_trajectory/flags, así que move_draw_sub no cambia.

Parámetros: imagen, ancho_mm, alto_mm (caja límite) y cualquier clave de
IMAGE_DEFAULTS.
"""
from __future__ import annotations

import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Int8MultiArray, MultiArrayDimension

from ur5_algoritmos.cinematica import TF2xyzquat, fkine_ur5
from ur5_algoritmos.trayectorias.fourier_image_functions import (
    IMAGE_DEFAULTS, run_fourier_image_pipeline,
)

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class FourierImagePublisher(Node):

    def __init__(self):
        super().__init__("fourier_image_trajectory")
        self.declare_parameter("imagen", "")
        self.declare_parameter("offset_x", 0.0)      # [m] desplazamiento del centro
        self.declare_parameter("offset_y", 0.0)
        self.declare_parameter("publish_repeats", 10)
        self.declare_parameter("publish_hz", 1.0)
        for name, value in IMAGE_DEFAULTS.items():
            self.declare_parameter(name, value)

        self.xy_pub = self.create_publisher(Float32MultiArray, "letter_trajectory/xy", 10)
        self.flag_pub = self.create_publisher(Int8MultiArray, "letter_trajectory/flags", 10)
        self.center = None
        self.done = False
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.get_logger().info("Esperando /joint_states para tomar la posición actual como centro...")

    def _on_joint_state(self, msg: JointState) -> None:
        if self.center is not None:
            return
        try:
            q = np.array([msg.position[msg.name.index(j)] for j in JOINT_NAMES])
        except ValueError:
            return
        x = TF2xyzquat(fkine_ur5(q))
        self.center = np.array([x[0] + self.get_parameter("offset_x").value,
                                x[1] + self.get_parameter("offset_y").value])
        self.get_logger().info(
            f"Centro del dibujo = posición actual: X={self.center[0]:.4f} Y={self.center[1]:.4f}")
        self._start()

    def _start(self) -> None:
        ruta = os.path.expanduser(self.get_parameter("imagen").value)
        if not ruta or not os.path.exists(ruta):
            self.get_logger().error(f"No existe la imagen: '{ruta}'")
            self.done = True
            return
        overrides = {k: self.get_parameter(k).value for k in IMAGE_DEFAULTS}
        t0 = time.perf_counter()
        trazos, data = run_fourier_image_pipeline(ruta, **overrides)
        dt = time.perf_counter() - t0

        largo = sum(t["largo_mm"] for t in trazos) / 1000.0
        n_wp = sum(len(t["points_mm"]) for t in trazos)
        self.get_logger().info(
            f"{os.path.basename(ruta)}: {len(trazos)} trazos (líneas {data['n_lineas']}, "
            f"rellenos {data['n_rellenos']}, sombreado {data['n_sombreado']}, "
            f"bordes {data['n_bordes']} antes de fusionar) | {largo:.2f} m | "
            f"{n_wp} waypoints | {dt:.1f} s")

        trayectorias = [self.center + t["points_mm"] / 1000.0 for t in trazos if len(t["points_mm"])]
        self._xy_msg, self._flag_msg = self._build_messages(trayectorias)
        self._count = 0
        self._repeats = max(1, int(self.get_parameter("publish_repeats").value))
        self._publish()
        self._timer = self.create_timer(
            1.0 / max(0.1, float(self.get_parameter("publish_hz").value)), self._timer_cb)

    def _timer_cb(self) -> None:
        if self._count >= self._repeats:
            self._timer.cancel()
            self.get_logger().info("Publicacion completada.")
            self.done = True
            return
        self._publish()

    def _publish(self) -> None:
        self._count += 1
        self.xy_pub.publish(self._xy_msg)
        self.flag_pub.publish(self._flag_msg)
        self.get_logger().info(f"Publicación {self._count}/{self._repeats}")

    @staticmethod
    def _build_messages(trayectorias: list):
        """Concatena las trayectorias; flags marca con 1 el inicio de cada trazo."""
        all_xy = np.concatenate(trayectorias, axis=0)
        flags = np.zeros(len(all_xy), dtype=np.int8)
        flags[np.cumsum([0] + [len(t) for t in trayectorias[:-1]])] = 1
        n = len(flags)
        xy_msg = Float32MultiArray()
        xy_msg.layout.dim = [
            MultiArrayDimension(label="points", size=n, stride=n * 2),
            MultiArrayDimension(label="xy", size=2, stride=2),
        ]
        xy_msg.data = all_xy.astype(np.float32).flatten().tolist()
        flag_msg = Int8MultiArray()
        flag_msg.layout.dim = [MultiArrayDimension(label="points", size=n, stride=n)]
        flag_msg.data = flags.tolist()
        return xy_msg, flag_msg


def main(args=None):
    rclpy.init(args=args)
    node = FourierImagePublisher()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
