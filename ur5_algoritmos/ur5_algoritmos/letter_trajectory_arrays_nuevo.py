#!/usr/bin/env python3
"""
array.py — Publicador ROS2 de trayectorias de letras usando pipeline V6 A4.

Versión sin RViz ni markers.

Flujo:
    texto -> imagen A4 -> binarización/marcas -> esqueleto -> caminos
    -> splines -> mm A4 -> XY robot

Publica:
    /letter_trajectory/xy      Float32MultiArray  [x0,y0, x1,y1, ..., xN,yN]
    /letter_trajectory/flags   Int8MultiArray     [1,0,0,...,1,0,0,...]
"""

from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int8MultiArray, MultiArrayDimension

# En ROS2 debe importar desde el paquete. El fallback permite probarlo localmente
# si array.py y letter_trajectory_functions_v6.py están en la misma carpeta.

from ur5_algoritmos.letter_trajectory_functions_v6 import (
        run_full_pipeline,
        A4_WIDTH_MM,
        A4_HEIGHT_MM,
        PX_PER_MM,
        IMAGE_SIZE as A4_IMAGE_SIZE,
        ROI_MM as DEFAULT_ROI_MM,
        MARGIN as DEFAULT_MARGIN,
    )



# =============================================================================
# Parámetros base
# =============================================================================

DEFAULT_TEXT = "Marko"
DEFAULT_OUTPUT_DIR = "letras_png_A4"

# Se buscan rutas comunes usadas en tus workspaces. Si ninguna existe,
# letter_trajectory_functions_v6.py usará una fuente .ttf/.otf de respaldo.
DEFAULT_FONT_CANDIDATES = [
    "/home/utec/ros2_ws/src/ur5_algoritmos/ur5_algoritmos/Playwrite_CU/PlaywriteCU-VariableFont_wght.ttf",
    "/home/utec/barirobot_ws/src/ur5_algoritmos/ur5_algoritmos/Playwrite_CU/PlaywriteCU-VariableFont_wght.ttf",
    "/home/utec/llm_ur5/V2/Tipografia1/PlaywriteCU-VariableFont_wght.ttf",
    os.path.join(os.path.dirname(__file__), "Playwrite_CU", "PlaywriteCU-VariableFont_wght.ttf"),
]

# Parámetros del pipeline V6
MAX_FONT_SIZE = 0  # 0 = None, se calcula automáticamente
THRESHOLD_VALUE = 200
CLOSE_KERNEL_SIZE = 5
CLOSE_ITERATIONS = 1
EROSION_KERNEL_SIZE = 3
EROSION_ITERATIONS = 0

DETECT_MARKS = True
MARK_MIN_AREA = 20
MARK_MAX_AREA = 1200
MARK_MIN_SIZE = 4
MARK_MAX_SIZE = 75
MARK_UPPER_REGION_QUANTILE = 0.72
DOT_MAX_ASPECT_RATIO = 1.65
DOT_MAX_ECCENTRICITY = 0.78
DIAERESIS_PAIR_MIN_DX = 8
DIAERESIS_PAIR_MAX_DX = 58
DIAERESIS_PAIR_MAX_DY = 18
DOT_SPLINE_POINTS = 70
DOT_RADIUS_SCALE = 0.85
ACCENT_MIN_PATH_PIXELS = 4

MIN_PATH_PIXELS = 12
ORDER_STRATEGY = "left_to_right"

MERGE_CLOSE_PATHS = True
MERGE_MAX_GAP_PIXELS = 30
MERGE_MAX_VERTICAL_GAP_PIXELS = 50
MERGE_CONNECTOR_POINTS = 8
MERGE_WITH_RETRACE = True
RETRACE_ATTACH_MAX_DISTANCE = 16
RETRACE_MAX_BACKTRACK_PIXELS = 75
RETRACE_CONNECTOR_POINTS = 4

SPLINE_SMOOTHING = 2.0
SPLINE_POINTS = 40
SPLINE_DEGREE = 3

# Mapeo A4 mm -> plano XY del robot.
# El ROI define la zona física de escritura dentro de la hoja A4.
DRAW_CENTER_X = -0.8
DRAW_CENTER_Y = 0.15
DRAW_WIDTH = 0.20
DRAW_HEIGHT = 0.10

# Publicación
PUBLISH_REPEATS = 10
PUBLISH_HZ = 1.0


# =============================================================================
# Utilidades
# =============================================================================

def choose_default_font_path() -> str:
    """Retorna la primera ruta de fuente existente, o la primera candidata."""
    for candidate in DEFAULT_FONT_CANDIDATES:
        if candidate and os.path.exists(candidate):
            return candidate
    return DEFAULT_FONT_CANDIDATES[0]


def mm_to_robot_xy(
    x_mm: float,
    y_mm: float,
    *,
    roi_mm: Sequence[float] = DEFAULT_ROI_MM,
    draw_center_x: float = DRAW_CENTER_X,
    draw_center_y: float = DRAW_CENTER_Y,
    draw_width: float = DRAW_WIDTH,
    draw_height: float = DRAW_HEIGHT,
) -> np.ndarray:
    """
    Convierte coordenadas [x_mm, y_mm] sobre la hoja A4 a [x, y] del robot.

    El eje Y se invierte porque en la imagen y_mm crece hacia abajo, mientras
    que en el plano de escritura del robot se toma positivo hacia arriba.
    """
    roi_x, roi_y, roi_w, roi_h = [float(v) for v in roi_mm]

    nx = ((float(x_mm) - roi_x) / roi_w) - 0.5
    ny = -(((float(y_mm) - roi_y) / roi_h) - 0.5)

    return np.array(
        [
            float(draw_center_x) + nx * float(draw_width),
            float(draw_center_y) + ny * float(draw_height),
        ],
        dtype=float,
    )


# =============================================================================
# Nodo ROS2
# =============================================================================

class LetterTrajectoryPublisher(Node):
    """Genera una palabra con pipeline V6 y publica la trayectoria como arrays."""

    def __init__(self):
        super().__init__("letter_trajectory_publisher_v6")

        self._declare_parameters()
        self._load_parameters()

        self.xy_pub = self.create_publisher(
            Float32MultiArray,
            "letter_trajectory/xy",
            10,
        )
        self.flag_pub = self.create_publisher(
            Int8MultiArray,
            "letter_trajectory/flags",
            10,
        )

        self.trajectories_xy, self.splines, self.data, self.graph_info = self._generate_trajectories()

        if not self.trajectories_xy:
            self.get_logger().error("No se generaron trayectorias.")
            return

        self._xy_msg, self._flag_msg = self._build_messages(self.trajectories_xy)

        n_total = sum(len(t) for t in self.trajectories_xy)
        self.get_logger().info(
            f"{n_total} waypoints en {len(self.trajectories_xy)} trazos — "
            f"publicando {self.publish_repeats}x a {self.publish_hz:.2f} Hz"
        )

        self._count = 0
        self._do_publish()
        self._timer = self.create_timer(1.0 / self.publish_hz, self._timer_cb)

    # ------------------------------------------------------------------
    # Parámetros ROS
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter("text", DEFAULT_TEXT)
        self.declare_parameter("font_path", choose_default_font_path())
        self.declare_parameter("output_dir", DEFAULT_OUTPUT_DIR)

        self.declare_parameter("max_font_size", MAX_FONT_SIZE)
        self.declare_parameter("threshold_value", THRESHOLD_VALUE)
        self.declare_parameter("close_kernel_size", CLOSE_KERNEL_SIZE)
        self.declare_parameter("close_iterations", CLOSE_ITERATIONS)
        self.declare_parameter("erosion_kernel_size", EROSION_KERNEL_SIZE)
        self.declare_parameter("erosion_iterations", EROSION_ITERATIONS)

        self.declare_parameter("detect_marks", DETECT_MARKS)
        self.declare_parameter("mark_min_area", MARK_MIN_AREA)
        self.declare_parameter("mark_max_area", MARK_MAX_AREA)
        self.declare_parameter("mark_min_size", MARK_MIN_SIZE)
        self.declare_parameter("mark_max_size", MARK_MAX_SIZE)
        self.declare_parameter("mark_upper_region_quantile", MARK_UPPER_REGION_QUANTILE)
        self.declare_parameter("dot_max_aspect_ratio", DOT_MAX_ASPECT_RATIO)
        self.declare_parameter("dot_max_eccentricity", DOT_MAX_ECCENTRICITY)
        self.declare_parameter("diaeresis_pair_min_dx", DIAERESIS_PAIR_MIN_DX)
        self.declare_parameter("diaeresis_pair_max_dx", DIAERESIS_PAIR_MAX_DX)
        self.declare_parameter("diaeresis_pair_max_dy", DIAERESIS_PAIR_MAX_DY)
        self.declare_parameter("dot_spline_points", DOT_SPLINE_POINTS)
        self.declare_parameter("dot_radius_scale", DOT_RADIUS_SCALE)
        self.declare_parameter("accent_min_path_pixels", ACCENT_MIN_PATH_PIXELS)

        self.declare_parameter("min_path_pixels", MIN_PATH_PIXELS)
        self.declare_parameter("order_strategy", ORDER_STRATEGY)

        self.declare_parameter("merge_close_paths", MERGE_CLOSE_PATHS)
        self.declare_parameter("merge_max_gap_pixels", MERGE_MAX_GAP_PIXELS)
        self.declare_parameter("merge_max_vertical_gap_pixels", MERGE_MAX_VERTICAL_GAP_PIXELS)
        self.declare_parameter("merge_connector_points", MERGE_CONNECTOR_POINTS)
        self.declare_parameter("merge_with_retrace", MERGE_WITH_RETRACE)
        self.declare_parameter("retrace_attach_max_distance", RETRACE_ATTACH_MAX_DISTANCE)
        self.declare_parameter("retrace_max_backtrack_pixels", RETRACE_MAX_BACKTRACK_PIXELS)
        self.declare_parameter("retrace_connector_points", RETRACE_CONNECTOR_POINTS)

        self.declare_parameter("spline_smoothing", SPLINE_SMOOTHING)
        self.declare_parameter("spline_points", SPLINE_POINTS)
        self.declare_parameter("spline_degree", SPLINE_DEGREE)

        self.declare_parameter("roi_x_mm", float(DEFAULT_ROI_MM[0]))
        self.declare_parameter("roi_y_mm", float(DEFAULT_ROI_MM[1]))
        self.declare_parameter("roi_w_mm", float(DEFAULT_ROI_MM[2]))
        self.declare_parameter("roi_h_mm", float(DEFAULT_ROI_MM[3]))
        self.declare_parameter("px_per_mm", float(PX_PER_MM))

        self.declare_parameter("draw_center_x", DRAW_CENTER_X)
        self.declare_parameter("draw_center_y", DRAW_CENTER_Y)
        self.declare_parameter("draw_width", DRAW_WIDTH)
        self.declare_parameter("draw_height", DRAW_HEIGHT)

        self.declare_parameter("publish_repeats", PUBLISH_REPEATS)
        self.declare_parameter("publish_hz", PUBLISH_HZ)

    def _load_parameters(self) -> None:
        self.text = str(self.get_parameter("text").value)
        self.font_path = str(self.get_parameter("font_path").value)
        self.output_dir = str(self.get_parameter("output_dir").value)

        max_font_size = int(self.get_parameter("max_font_size").value)
        self.max_font_size = None if max_font_size <= 0 else max_font_size

        self.threshold_value = int(self.get_parameter("threshold_value").value)
        self.close_kernel_size = int(self.get_parameter("close_kernel_size").value)
        self.close_iterations = int(self.get_parameter("close_iterations").value)
        self.erosion_kernel_size = int(self.get_parameter("erosion_kernel_size").value)
        self.erosion_iterations = int(self.get_parameter("erosion_iterations").value)

        self.detect_marks = bool(self.get_parameter("detect_marks").value)
        self.mark_min_area = int(self.get_parameter("mark_min_area").value)
        self.mark_max_area = int(self.get_parameter("mark_max_area").value)
        self.mark_min_size = int(self.get_parameter("mark_min_size").value)
        self.mark_max_size = int(self.get_parameter("mark_max_size").value)
        self.mark_upper_region_quantile = float(self.get_parameter("mark_upper_region_quantile").value)
        self.dot_max_aspect_ratio = float(self.get_parameter("dot_max_aspect_ratio").value)
        self.dot_max_eccentricity = float(self.get_parameter("dot_max_eccentricity").value)
        self.diaeresis_pair_min_dx = int(self.get_parameter("diaeresis_pair_min_dx").value)
        self.diaeresis_pair_max_dx = int(self.get_parameter("diaeresis_pair_max_dx").value)
        self.diaeresis_pair_max_dy = int(self.get_parameter("diaeresis_pair_max_dy").value)
        self.dot_spline_points = int(self.get_parameter("dot_spline_points").value)
        self.dot_radius_scale = float(self.get_parameter("dot_radius_scale").value)
        self.accent_min_path_pixels = int(self.get_parameter("accent_min_path_pixels").value)

        self.min_path_pixels = int(self.get_parameter("min_path_pixels").value)
        self.order_strategy = str(self.get_parameter("order_strategy").value)

        self.merge_close_paths = bool(self.get_parameter("merge_close_paths").value)
        self.merge_max_gap_pixels = int(self.get_parameter("merge_max_gap_pixels").value)
        self.merge_max_vertical_gap_pixels = int(self.get_parameter("merge_max_vertical_gap_pixels").value)
        self.merge_connector_points = int(self.get_parameter("merge_connector_points").value)
        self.merge_with_retrace = bool(self.get_parameter("merge_with_retrace").value)
        self.retrace_attach_max_distance = int(self.get_parameter("retrace_attach_max_distance").value)
        self.retrace_max_backtrack_pixels = int(self.get_parameter("retrace_max_backtrack_pixels").value)
        self.retrace_connector_points = int(self.get_parameter("retrace_connector_points").value)

        self.spline_smoothing = float(self.get_parameter("spline_smoothing").value)
        self.spline_points = int(self.get_parameter("spline_points").value)
        self.spline_degree = int(self.get_parameter("spline_degree").value)

        self.roi_mm = (
            float(self.get_parameter("roi_x_mm").value),
            float(self.get_parameter("roi_y_mm").value),
            float(self.get_parameter("roi_w_mm").value),
            float(self.get_parameter("roi_h_mm").value),
        )
        self.px_per_mm = float(self.get_parameter("px_per_mm").value)

        self.draw_center_x = float(self.get_parameter("draw_center_x").value)
        self.draw_center_y = float(self.get_parameter("draw_center_y").value)
        self.draw_width = float(self.get_parameter("draw_width").value)
        self.draw_height = float(self.get_parameter("draw_height").value)

        self.publish_repeats = max(1, int(self.get_parameter("publish_repeats").value))
        self.publish_hz = max(0.1, float(self.get_parameter("publish_hz").value))

    # ------------------------------------------------------------------
    # Publicación periódica
    # ------------------------------------------------------------------

    def _timer_cb(self) -> None:
        if self._count >= self.publish_repeats:
            self._timer.cancel()
            self.get_logger().info("Publicación completada.")
            return
        self._do_publish()

    def _do_publish(self) -> None:
        self._count += 1
        self.xy_pub.publish(self._xy_msg)
        self.flag_pub.publish(self._flag_msg)
        self.get_logger().info(f"Publicación {self._count}/{self.publish_repeats}")

    # ------------------------------------------------------------------
    # Pipeline V6 completo
    # ------------------------------------------------------------------

    def _generate_trajectories(self):
        """
        Ejecuta run_full_pipeline() de V6 y convierte splines A4 mm a XY robot.
        """
        self.get_logger().info(f"Texto: '{self.text}'")
        self.get_logger().info(f"Fuente: {self.font_path}")
        self.get_logger().info(
            f"A4: {A4_WIDTH_MM:.1f}x{A4_HEIGHT_MM:.1f} mm | "
            f"ROI: x={self.roi_mm[0]}, y={self.roi_mm[1]}, "
            f"w={self.roi_mm[2]}, h={self.roi_mm[3]} mm"
        )

        splines, spline_arrays_mm, data, graph_info = run_full_pipeline(
            text=self.text,
            font_path=self.font_path,
            output_dir=self.output_dir,
            image_size=A4_IMAGE_SIZE,
            max_font_size=self.max_font_size,
            margin=DEFAULT_MARGIN,
            roi_mm=self.roi_mm,
            px_per_mm=self.px_per_mm,
            threshold_value=self.threshold_value,
            close_kernel_size=self.close_kernel_size,
            close_iterations=self.close_iterations,
            erosion_kernel_size=self.erosion_kernel_size,
            erosion_iterations=self.erosion_iterations,
            detect_marks=self.detect_marks,
            mark_min_area=self.mark_min_area,
            mark_max_area=self.mark_max_area,
            mark_min_size=self.mark_min_size,
            mark_max_size=self.mark_max_size,
            mark_upper_region_quantile=self.mark_upper_region_quantile,
            dot_max_aspect_ratio=self.dot_max_aspect_ratio,
            dot_max_eccentricity=self.dot_max_eccentricity,
            diaeresis_pair_min_dx=self.diaeresis_pair_min_dx,
            diaeresis_pair_max_dx=self.diaeresis_pair_max_dx,
            diaeresis_pair_max_dy=self.diaeresis_pair_max_dy,
            dot_radius_scale=self.dot_radius_scale,
            dot_spline_points=self.dot_spline_points,
            accent_min_path_pixels=self.accent_min_path_pixels,
            min_path_pixels=self.min_path_pixels,
            order_strategy=self.order_strategy,
            merge_close_paths=self.merge_close_paths,
            merge_max_gap_pixels=self.merge_max_gap_pixels,
            merge_max_vertical_gap_pixels=self.merge_max_vertical_gap_pixels,
            merge_connector_points=self.merge_connector_points,
            merge_with_retrace=self.merge_with_retrace,
            retrace_attach_max_distance=self.retrace_attach_max_distance,
            retrace_max_backtrack_pixels=self.retrace_max_backtrack_pixels,
            retrace_connector_points=self.retrace_connector_points,
            spline_smoothing=self.spline_smoothing,
            spline_points=self.spline_points,
            spline_degree=self.spline_degree,
        )

        self._log_pipeline_stats(splines, data, graph_info)

        trajectories_xy = []
        for points_mm in spline_arrays_mm:
            if len(points_mm) == 0:
                continue
            traj_xy = np.array(
                [
                    mm_to_robot_xy(
                        x_mm,
                        y_mm,
                        roi_mm=self.roi_mm,
                        draw_center_x=self.draw_center_x,
                        draw_center_y=self.draw_center_y,
                        draw_width=self.draw_width,
                        draw_height=self.draw_height,
                    )
                    for x_mm, y_mm in points_mm
                ],
                dtype=float,
            )
            trajectories_xy.append(traj_xy)

        return trajectories_xy, splines, data, graph_info

    def _log_pipeline_stats(self, splines, data, graph_info) -> None:
        mark_counts = {}
        for comp in data.get("mark_components", []):
            kind = comp.get("mark_kind", "unknown")
            mark_counts[kind] = mark_counts.get(kind, 0) + 1

        self.get_logger().info(
            f"Marcas detectadas: puntos={mark_counts.get('dot_i_j', 0)}, "
            f"diéresis={mark_counts.get('diaeresis_dot', 0)}, "
            f"tildes/acentos={mark_counts.get('accent_or_tilde', 0)}"
        )
        self.get_logger().info(
            f"Grafo: extremos={len(graph_info.get('endpoints', []))} | "
            f"bifurcaciones={len(graph_info.get('junctions', []))}"
        )
        self.get_logger().info(f"Splines generados: {len(splines)}")

    # ------------------------------------------------------------------
    # Construcción de mensajes
    # ------------------------------------------------------------------

    def _build_messages(self, trajectories_xy: List[np.ndarray]):
        all_xy = np.concatenate(trajectories_xy, axis=0)
        xy_flat = all_xy.astype(np.float32).flatten()

        flags = np.zeros(len(all_xy), dtype=np.int8)
        idx = 0
        for traj in trajectories_xy:
            flags[idx] = 1
            idx += len(traj)

        n = len(flags)

        xy_msg = Float32MultiArray()
        xy_msg.layout.dim = [
            MultiArrayDimension(label="points", size=n, stride=n * 2),
            MultiArrayDimension(label="xy", size=2, stride=2),
        ]
        xy_msg.data = xy_flat.tolist()

        flag_msg = Int8MultiArray()
        flag_msg.layout.dim = [
            MultiArrayDimension(label="points", size=n, stride=n),
        ]
        flag_msg.data = flags.tolist()

        return xy_msg, flag_msg


# =============================================================================
# Entry point
# =============================================================================

def main(args=None) -> None:
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
