#!/usr/bin/env python3
"""
letter_trajectory_publisher.py

Publishes the full trajectory as two arrays, repeated PUBLISH_REPEATS times
at PUBLISH_HZ so late-joining subscribers don't miss it.

    /letter_trajectory/xy      (Float32MultiArray)  [x0,y0, x1,y1, ..., xN,yN]
    /letter_trajectory/flags   (Int8MultiArray)      [1,0,0,...,1,0,0,...]
                                                       ^ new segment
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int8MultiArray, MultiArrayDimension
import matplotlib.pyplot as plt

from ur5_algoritmos.letter_trajectory_functions import (
    analyze_skeleton_points,
    merge_segments_human_writing_order,
    plot_letter_pil,
    process_image,
    segments_to_spline_trajectories,
    skeleton_to_ordered_segments,
)

# ===========================================================================
# Parámetros
# ===========================================================================

FONTS = {
    "1": (
        "Playwrite CU",
        "/home/utec/ros2_ws/src/ur5_algoritmos/ur5_algoritmos/Playwrite_CU/PlaywriteCU-VariableFont_wght.ttf",
    ),
}

LETTER   = "Gianmarco"
FONT_KEY = "1"

MERGE_MAX_DIST            = 28.0
MERGE_MAX_ANGLE           = 125.0
MERGE_BRIDGE_PTS          = 5
MERGE_CYCLE_DIST          = 24.0
MERGE_EXPECTED_LETTER_GAP = None

SPLINE_SMOOTHING = 1.5
SPLINE_NUM_PTS   = 60

IMAGE_SIZE  = 800
DRAW_CENTER = np.array([-0.70, 0.0])
DRAW_WIDTH  = 0.15
DRAW_HEIGHT = 0.28

PUBLISH_REPEATS = 10     # how many times to send both arrays
PUBLISH_HZ      = 1.0   # rate between repetitions


# ===========================================================================
# Utilidad
# ===========================================================================
def rotate_point_z90_ccw_on_center(point, center=DRAW_CENTER):
    # Trasladar al origen
    p_translated = point - center
    # Rotar 90 grados antihorario
    p_rotated = np.array([-p_translated[1], p_translated[0]])
    # Devolver al centro original
    return p_rotated + center
    
def pixel_to_robot_xy(px, py, img_size=IMAGE_SIZE):
    nx =  (px / img_size) - 0.5
    ny = -((py / img_size) - 0.5)
    return np.array([
        DRAW_CENTER[0] + nx * DRAW_WIDTH,
        DRAW_CENTER[1] + ny * DRAW_HEIGHT,
    ])


# ===========================================================================
# Nodo
# ===========================================================================

class LetterTrajectoryPublisher(Node):

    def __init__(self):
        super().__init__("letter_trajectory_publisher")

        self.xy_pub   = self.create_publisher(Float32MultiArray, "letter_trajectory/xy",    10)
        self.flag_pub = self.create_publisher(Int8MultiArray,    "letter_trajectory/flags", 10)

        trajectories = self._generate_trajectories()
        if not trajectories:
            self.get_logger().error("No se generaron trayectorias.")
            return
        
        
        # Build messages once, reuse on every publish
        self._xy_msg, self._flag_msg = self._build_messages(trajectories)

        n_total = sum(len(t) for t in trajectories)
        self.get_logger().info(
            f"{n_total} waypoints en {len(trajectories)} segmentos — "
            f"publicando {PUBLISH_REPEATS}x a {PUBLISH_HZ} Hz"
        )
   
        self._count = 0
        #self._plot_published_data()
        self._do_publish()   # publish immediately, then repeat via timer
        
        self._timer = self.create_timer(1.0 / PUBLISH_HZ, self._timer_cb)

    # -----------------------------------------------------------------------

    def _timer_cb(self):
        if self._count >= PUBLISH_REPEATS:
            self._timer.cancel()
            self.get_logger().info("Publicacion completada.")
            return
        self._do_publish()

    def _do_publish(self):
        self._count += 1
        self.xy_pub.publish(self._xy_msg)
        self.flag_pub.publish(self._flag_msg)
        self.get_logger().info(f"Publicacion {self._count}/{PUBLISH_REPEATS}")

    # -----------------------------------------------------------------------
    # Pipeline V8
    # -----------------------------------------------------------------------

    def _generate_trajectories(self):
        font_name, font_path = FONTS[FONT_KEY]
        self.get_logger().info(f"Texto: '{LETTER}' | Fuente: {font_name}")

        file_path = plot_letter_pil(LETTER, font_path)
        _, _, skeleton, _ = process_image(file_path)
        if skeleton is None:
            self.get_logger().error("Sin esqueleto")
            return []

        endpoints, junctions, _ = analyze_skeleton_points(skeleton)
        self.get_logger().info(f"Extremos: {len(endpoints)} | Bifurcaciones: {len(junctions)}")

        segments = skeleton_to_ordered_segments(
            skeleton, min_length=3, junction_dilate=2, keep_cycles=True,
        )
        merged = merge_segments_human_writing_order(
            segments,
            max_endpoint_distance     = MERGE_MAX_DIST,
            max_angle                 = MERGE_MAX_ANGLE,
            bridge_points             = MERGE_BRIDGE_PTS,
            min_branch_length         = 0,
            keep_closed_cycles        = True,
            open_closed_cycles        = True,
            cycle_connection_distance = MERGE_CYCLE_DIST,
            protect_short_isolated    = True,
            expected_letter_gap       = MERGE_EXPECTED_LETTER_GAP,
            verbose                   = True,
        )
        trajectories_px = segments_to_spline_trajectories(
            merged, smoothing=SPLINE_SMOOTHING, num_points=SPLINE_NUM_PTS,
        )
        return [
            np.array([rotate_point_z90_ccw_on_center(pixel_to_robot_xy(px, py)) for px, py in traj])
            #np.array([pixel_to_robot_xy(px, py) for px, py in traj])
            for traj in trajectories_px
        ]

    # -----------------------------------------------------------------------
    # Build flat arrays
    # -----------------------------------------------------------------------

    def _build_messages(self, trajectories):
        all_xy  = np.concatenate(trajectories, axis=0)          # (N, 2)
        xy_flat = all_xy.flatten().astype(np.float32)           # (2N,)

        flags = np.zeros(len(all_xy), dtype=np.int8)
        idx = 0
        for traj in trajectories:
            flags[idx] = 1
            idx += len(traj)

        n = len(flags)

        xy_msg = Float32MultiArray()
        xy_msg.layout.dim = [
            MultiArrayDimension(label="points", size=n, stride=n * 2),
            MultiArrayDimension(label="xy",     size=2, stride=2),
        ]
        xy_msg.data = xy_flat.tolist()

        flag_msg = Int8MultiArray()
        flag_msg.layout.dim = [
            MultiArrayDimension(label="points", size=n, stride=n),
        ]
        flag_msg.data = flags.tolist()

        return xy_msg, flag_msg
        
    def _plot_trajectories(self, trajectories):
        plt.figure(figsize=(8, 8))

        for i, traj in enumerate(trajectories):
            plt.plot(
            traj[:, 0],
            traj[:, 1],
            '-',
            linewidth=2,
            label=f'Segmento {i+1}'
        )

        # Marcar inicio del segmento
            plt.plot(
            traj[0, 0],
            traj[0, 1],
            'go',
            markersize=8
        )

        # Marcar final del segmento
            plt.plot(
            traj[-1, 0],
            traj[-1, 1],
            'ro',
            markersize=8
        )

        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title(f"Trayectoria: {LETTER}")

        # Mostrar ejes positivos y negativos
        plt.axhline(y=0, linestyle='--')
        plt.axvline(x=0, linestyle='--')

        plt.grid(True)
        plt.axis('equal')

        plt.legend()
        plt.show(block=False)
    
    def _plot_published_data(self):

        self.get_logger().info("Graficando trayectoria...")
        data = np.array(self._xy_msg.data)

        x = data[0::2]
        y = data[1::2]

        plt.figure(figsize=(8,8))
        plt.plot(x, y, 'o', markersize=2)

        plt.axhline(0, linestyle='--')
        plt.axvline(0, linestyle='--')

        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Puntos publicados")
        plt.grid(True)
        plt.axis('equal')

        plt.show()


# ===========================================================================
# Entry point
# ===========================================================================

def main(args=None):
    rclpy.init(args=args)
    node = LetterTrajectoryPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
