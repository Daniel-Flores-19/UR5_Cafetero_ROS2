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
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Int8MultiArray, MultiArrayDimension
import matplotlib.pyplot as plt

from ur5_algoritmos.cinematica import TF2xyzquat, fkine_ur5
from ur5_algoritmos.trayectorias.painting_trajectory_functions import get_polygons, scanline_fill, resample_polyline

# ===========================================================================
# Parámetros
# ===========================================================================

# El centro del dibujo ya no es fijo: es el XY donde está el efector al
# arrancar el nodo (primer /joint_states + FK). Basta con dejar el plumón sobre
# el punto donde se quiere la imagen.
# DRAW_WIDTH x DRAW_HEIGHT es la caja LÍMITE: la imagen se escala con un único
# factor (el mayor que cabe), así que no se deforma aunque no sea cuadrada.
DRAW_WIDTH  = 0.20
DRAW_HEIGHT = 0.20

PUBLISH_REPEATS = 10     # how many times to send both arrays
PUBLISH_HZ      = 1.0   # rate between repetitions

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


# ===========================================================================
# Utilidad
# ===========================================================================

def pixel_to_robot_xy(px, py, img_size, center):
    # Misma escala [m/px] en X e Y: la proporción de la imagen se conserva.
    scale = min(DRAW_WIDTH / img_size[1], DRAW_HEIGHT / img_size[0])
    dx =  (px - img_size[1] / 2.0) * scale
    dy = -(py - img_size[0] / 2.0) * scale
    return np.array([center[0] + dx, center[1] + dy])


# ===========================================================================
# Nodo
# ===========================================================================

class LetterTrajectoryPublisher(Node):

    def __init__(self):
        super().__init__("letter_trajectory_publisher")

        self.xy_pub   = self.create_publisher(Float32MultiArray, "letter_trajectory/xy",    10)
        self.flag_pub = self.create_publisher(Int8MultiArray,    "letter_trajectory/flags", 10)

        self.done = False
        self.center = None
        self._js_sub = self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.get_logger().info("Esperando /joint_states para tomar la posición actual como centro...")

    # -----------------------------------------------------------------------

    def _joint_state_cb(self, msg):
        """Toma UNA vez el XY actual del efector como centro y arranca."""
        if self.center is not None:
            return
        try:
            q = np.array([msg.position[msg.name.index(j)] for j in JOINT_NAMES], dtype=float)
        except ValueError:
            return
        x = np.asarray(TF2xyzquat(fkine_ur5(q)), dtype=float)
        self.center = x[:2].copy()
        self.destroy_subscription(self._js_sub)
        self.get_logger().info(
            f"Centro de dibujo = posición actual del robot: "
            f"X={self.center[0]:.4f} Y={self.center[1]:.4f} (Z={x[2]:.4f})")
        self._start()

    def _start(self):
        trajectories = self._generate_trajectories()
        if not trajectories:
            self.get_logger().error("No se generaron trayectorias.")
            return
        
        
        # Build messages once, reuse on every publish
        self._xy_msg, self._flag_msg = self._build_messages(trajectories)


        # trajectories = [grupo cian, grupo negro]; cada grupo es una lista de trazos.
        trazos = [t for group in trajectories for t in group]
        n_total = sum(len(t) for t in trazos)
        self.get_logger().info(
            f"{n_total} waypoints en {len(trazos)} trazos ({len(trajectories)} colores) — "
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
            self.done = True 
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
        img = cv2.imread('/home/mito/Downloads/utec_test.jpg')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        black_mask = cv2.inRange(img, (0, 0, 0), (50, 50, 50))
        cyan_mask  = cv2.inRange(hsv, (85, 100, 100), (105, 255, 255))

        kernel = np.ones((3, 3), np.uint8)
        cyan_mask  = cv2.morphologyEx(cyan_mask,  cv2.MORPH_CLOSE, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

        cyan_polys  = get_polygons(cyan_mask)
        black_polys = get_polygons(black_mask)

        cyan_strokes  = scanline_fill(cyan_mask,  spacing=105)
        black_strokes = scanline_fill(black_mask, spacing=105)

        MAX_SPACING = 4.0

        cyan_traj = (
            [resample_polyline(p, MAX_SPACING, closed=True) for p in cyan_polys]
            #+ [resample_polyline([(x1, y1), (x2, y2)], MAX_SPACING, closed=False)
            #for (x1, y1), (x2, y2) in cyan_strokes]
        )
        black_traj = (
            [resample_polyline(p, MAX_SPACING, closed=True) for p in black_polys]
            #+ [resample_polyline([(x1, y1), (x2, y2)], MAX_SPACING, closed=False)
            #for (x1, y1), (x2, y2) in black_strokes]
        )

        scale = min(DRAW_WIDTH / img.shape[1], DRAW_HEIGHT / img.shape[0])
        self.get_logger().info(
            f"Imagen {img.shape[1]}x{img.shape[0]} px -> "
            f"{img.shape[1] * scale * 1000:.1f} x {img.shape[0] * scale * 1000:.1f} mm "
            f"(escala {scale * 1000:.4f} mm/px, caja {DRAW_WIDTH * 1000:.0f} x {DRAW_HEIGHT * 1000:.0f} mm)")

        color_groups = [
            [np.array([pixel_to_robot_xy(px, py, img.shape, self.center) for px, py in t]) for t in cyan_traj],
            [np.array([pixel_to_robot_xy(px, py, img.shape, self.center) for px, py in t]) for t in black_traj],
        ]
        return color_groups

    # -----------------------------------------------------------------------
    # Build flat arrays
    # -----------------------------------------------------------------------

    def _build_messages(self, color_groups):
        all_trajs = [traj for group in color_groups for traj in group]
        all_xy  = np.concatenate(all_trajs, axis=0)
        xy_flat = all_xy.flatten().astype(np.float32)

        flags = np.zeros(len(all_xy), dtype=np.int8)
        idx = 0
        for g, group in enumerate(color_groups):
            for t, traj in enumerate(group):
                flags[idx] = 2 if (t == 0 and g > 0) else 1
                idx += len(traj)

        n = len(flags)
        xy_msg = Float32MultiArray()
        xy_msg.layout.dim = [
            MultiArrayDimension(label="points", size=n, stride=n * 2),
            MultiArrayDimension(label="xy",     size=2, stride=2),
        ]
        xy_msg.data = xy_flat.tolist()

        flag_msg = Int8MultiArray()
        flag_msg.layout.dim = [MultiArrayDimension(label="points", size=n, stride=n)]
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
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
