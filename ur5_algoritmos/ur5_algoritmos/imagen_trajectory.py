#!/usr/bin/env python3
"""
letter_trajectory_publisher.py

Publishes the full trajectory as two arrays, repeated PUBLISH_REPEATS times
at PUBLISH_HZ so late-joining subscribers don't miss it.

    /letter_trajectory/xy      (Float32MultiArray)  [x0,y0, x1,y1, ..., xN,yN]
    /letter_trajectory/flags   (Int8MultiArray)      [1,0,0,...,1,0,0,...]
                                                       ^ new segment
"""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int8MultiArray, MultiArrayDimension
import matplotlib.pyplot as plt

from ur5_algoritmos.letter_trajectory_functions_img import (
    analyze_skeleton_points,
    merge_segments_human_writing_order,
    plot_letter_pil,
    process_image,
    segments_to_spline_trajectories,
    skeleton_to_ordered_segments,
    ur5_resize_paper,
    auto_canny_limits,
    merge_segments_by_best_continuation,
    filter_canny_by_density,
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

LETTER   = "Kai'sa"
FONT_KEY = "1"

MERGE_MAX_DIST            = 28.0
MERGE_MAX_ANGLE           = 125.0
MERGE_BRIDGE_PTS          = 5
MERGE_CYCLE_DIST          = 24.0
MERGE_EXPECTED_LETTER_GAP = None

SPLINE_SMOOTHING = 1.5
SPLINE_NUM_PTS   = 70


DRAW_CENTER = np.array([-0.70, 0.20])
DRAW_WIDTH  = 0.25
DRAW_HEIGHT = 0.35

PUBLISH_REPEATS = 50     # how many times to send both arrays
PUBLISH_HZ      = 1.0   # rate between repetitions


# ===========================================================================
# Utilidad
# ===========================================================================

def pixel_to_robot_xy(px, py, img_size):
    nx =  (px / img_size[1]) - 0.5
    ny = -((py / img_size[0]) - 0.5)
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
        self._plot_published_data()
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

        # Processing an image
        image = cv2.imread('/home/utec/Downloads/cartoonified_huggingface(2).png')
        img_resized = ur5_resize_paper(image)
        img_canny = auto_canny_limits(img_resized)
        

        img_canny_f = filter_canny_by_density(
            img_canny,
            window_size=31,
            density_threshold=0.05,
            suppression_mode="thin",
            )
        _, _, skeleton, _ = process_image(img_canny)
        if skeleton is None:
            self.get_logger().error("Sin esqueleto")
            return []

        endpoints, junctions, _ = analyze_skeleton_points(skeleton)
        self.get_logger().info(f"Extremos: {len(endpoints)} | Bifurcaciones: {len(junctions)}")

        segments = skeleton_to_ordered_segments(
            skeleton, min_length=15, junction_dilate=2, keep_cycles=True,
        )
        merged = merge_segments_by_best_continuation(
            segments,
            max_endpoint_distance = 8.0,   # much tighter — image edges are dense
            max_angle             = 60.0,  # stricter angle — follow contours, don't jump
            bridge_points         = 3,
            keep_closed_cycles    = True,
            open_closed_cycles    = True,
            )

        trajectories_px = segments_to_spline_trajectories(
            merged, smoothing=SPLINE_SMOOTHING, num_points=SPLINE_NUM_PTS,
        )
        return [
            np.array([pixel_to_robot_xy(px, py, img_canny.shape) for px, py in traj])
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
