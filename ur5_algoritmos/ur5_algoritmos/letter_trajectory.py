#!/usr/bin/env python3
"""
letter_trajectory.py — Publicador ROS2 de trayectorias de letras (pipeline V7 A4).

Flujo:
    texto -> imagen A4 -> binarización/marcas -> esqueleto -> caminos
    -> orden/fusión V7 -> splines -> mm A4 -> XY robot

Publica (tras recibir /image_drawing/done):
    /letter_trajectory/xy      Float32MultiArray  [x0,y0, x1,y1, ..., xN,yN]
    /letter_trajectory/flags   Int8MultiArray     [1,0,0,...,1,0,0,...]

Los parámetros ROS son los de DEFAULTS del pipeline (ver
letter_trajectory_functions.py) más los propios del nodo en NODE_DEFAULTS.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, Int8MultiArray, MultiArrayDimension

# En ROS2 se importa desde el paquete. El fallback permite probar el nodo
# localmente si ambos archivos están en la misma carpeta.
try:
    from ur5_algoritmos.trayectorias.letter_trajectory_functions import (
        A4_HEIGHT_MM, A4_WIDTH_MM, DEFAULTS, run_full_pipeline,
    )
except ImportError:  # pragma: no cover - solo para pruebas fuera del paquete ROS
    from letter_trajectory_functions import (
        A4_HEIGHT_MM, A4_WIDTH_MM, DEFAULTS, run_full_pipeline,
    )


# Tipografía del paquete (Tipografia/). Por defecto Ladislav Light, la fuente
# UTEC: es monolínea, así que el esqueleto cae sobre el trazo real y deja muchas
# menos bifurcaciones que una cursiva. Playwrite CU queda como respaldo. Si no
# aparece ninguna, el pipeline cae a una TrueType del sistema.
LADISLAV_LIGHT = os.path.join(
    "UTEC", "Ladislav-20260910T201106Z-1-001", "Ladislav", "LadislavLight.otf")
PLAYWRITE_TTF = os.path.join("Playwrite_CU", "PlaywriteCU-VariableFont_wght.ttf")

# Raíces donde buscar, en orden: la del propio paquete (instalado o en src) y
# los workspaces antiguos, que guardaban las familias sin la carpeta Tipografia.
TIPOGRAFIA_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tipografia"),
    "/home/mito/pintorV2_ws/src/ur5_algoritmos/ur5_algoritmos/Tipografia",
    "/home/mito/pintor_ws/src/ur5_algoritmos/ur5_algoritmos/Tipografia",
    "/home/utec/ros2_ws/src/ur5_algoritmos/ur5_algoritmos",
    "/home/utec/barirobot_ws/src/ur5_algoritmos/ur5_algoritmos",
]

# Se agota Ladislav en todas las raíces antes de probar con Playwrite.
FONT_CANDIDATES = [
    os.path.join(raiz, rel)
    for rel in (LADISLAV_LIGHT, PLAYWRITE_TTF)
    for raiz in TIPOGRAFIA_DIRS
]


def choose_default_font_path() -> str:
    """Primera fuente candidata existente, o la primera de la lista."""
    return next((p for p in FONT_CANDIDATES if p and os.path.exists(p)), FONT_CANDIDATES[0])


# Parámetros propios del nodo, más los del pipeline que se sobrescriben aquí.
NODE_DEFAULTS = {
    "text": "Faker",
    "font_path": choose_default_font_path(),
    "spline_points": 220,
    # Mapeo del ROI A4 al plano XY del robot [m]. draw_width y draw_height son
    # la caja LÍMITE del dibujo: se usa una sola escala para X e Y (la mayor que
    # cabe dentro), así que el texto ocupa una de las dos dimensiones y la otra
    # queda holgada, pero sin deformarse.
    "draw_center_x": -0.78,
    "draw_center_y": -0.26,
    "draw_width": 0.20,
    "draw_height": 0.09,
    # Publicación
    "publish_repeats": 10,
    "publish_hz": 1.0,
}
PARAM_DEFAULTS = {**DEFAULTS, **NODE_DEFAULTS}


class LetterTrajectoryPublisher(Node):
    """Genera una palabra con el pipeline V7 y publica su trayectoria como arrays."""

    def __init__(self):
        super().__init__("letter_trajectory_publisher_v7")

        for name, value in PARAM_DEFAULTS.items():
            self.declare_parameter(name, value)
        self.p = SimpleNamespace(**{name: self.get_parameter(name).value for name in PARAM_DEFAULTS})
        self.p.publish_repeats = max(1, int(self.p.publish_repeats))
        self.p.publish_hz = max(0.1, float(self.p.publish_hz))

        self.xy_pub = self.create_publisher(Float32MultiArray, "letter_trajectory/xy", 10)
        self.flag_pub = self.create_publisher(Int8MultiArray, "letter_trajectory/flags", 10)

        self.done = False
        self._started = False
        self.create_subscription(Bool, "image_drawing/done", self._on_image_done, 10)
        self.get_logger().info("Esperando señal de fin de dibujo de imagen...")

    # ------------------------------------------------------------------
    # Arranque y publicación periódica
    # ------------------------------------------------------------------

    def _on_image_done(self, msg: Bool) -> None:
        if not msg.data or self._started:
            return
        self._started = True
        self.get_logger().info("Señal recibida — iniciando pipeline de texto.")

        trajectories_xy = self._generate_trajectories()
        if not trajectories_xy:
            self.get_logger().error("No se generaron trayectorias.")
            return

        self._xy_msg, self._flag_msg = self._build_messages(trajectories_xy)
        self._count = 0
        self._publish()
        self._timer = self.create_timer(1.0 / self.p.publish_hz, self._timer_cb)

    def _timer_cb(self) -> None:
        if self._count >= self.p.publish_repeats:
            self._timer.cancel()
            self.get_logger().info("Publicacion completada.")
            self.done = True
            return
        self._publish()

    def _publish(self) -> None:
        self._count += 1
        self.xy_pub.publish(self._xy_msg)
        self.flag_pub.publish(self._flag_msg)
        self.get_logger().info(f"Publicación {self._count}/{self.p.publish_repeats}")

    # ------------------------------------------------------------------
    # Pipeline V7 -> trayectorias XY del robot
    # ------------------------------------------------------------------

    def _generate_trajectories(self) -> list:
        """Ejecuta el pipeline y pasa cada spline de mm A4 a XY del robot."""
        self.get_logger().info(f"Texto: '{self.p.text}' | Fuente: {self.p.font_path}")
        self.get_logger().info(
            f"A4: {A4_WIDTH_MM:.1f}x{A4_HEIGHT_MM:.1f} mm | "
            f"ROI: x={self.p.roi_x_mm}, y={self.p.roi_y_mm}, "
            f"w={self.p.roi_w_mm}, h={self.p.roi_h_mm} mm | "
            f"margen={self.p.roi_margin_mm:.2f} mm"
        )
        scale = self._draw_scale()
        self.get_logger().info(
            f"Escala hoja->robot: {scale * 1000.0:.3f} mm/mm | "
            f"dibujo: {self.p.roi_w_mm * scale * 1000.0:.1f}x"
            f"{self.p.roi_h_mm * scale * 1000.0:.1f} mm "
            f"dentro de {self.p.draw_width * 1000.0:.1f}x{self.p.draw_height * 1000.0:.1f} mm"
        )

        splines, spline_arrays_mm, data, graph_info = run_full_pipeline(
            self.p.text, self.p.font_path,
            **{key: getattr(self.p, key) for key in DEFAULTS},
        )
        self._log_pipeline_stats(splines, data, graph_info)

        return [self._mm_to_robot_xy(points_mm) for points_mm in spline_arrays_mm if len(points_mm)]

    def _draw_scale(self) -> float:
        """
        Escala única hoja -> robot [m/mm]: la mayor que cabe en la caja de dibujo.

        Al usar el mismo factor en X e Y el texto nunca se deforma, aunque el ROI
        y la caja (draw_width x draw_height) tengan proporciones distintas.
        """
        return min(self.p.draw_width / self.p.roi_w_mm,
                   self.p.draw_height / self.p.roi_h_mm)

    def _mm_to_robot_xy(self, points_mm: np.ndarray) -> np.ndarray:
        """
        Convierte puntos [x_mm, y_mm] de la hoja A4 a [x, y] del robot.

        El ROI se centra en (draw_center_x, draw_center_y) y se escala con
        _draw_scale(), igual en ambos ejes: se conserva la proporción que el
        pipeline ya respeta al pasar de píxeles a mm.

        El eje Y se invierte: en la imagen y_mm crece hacia abajo, mientras que
        en el plano de escritura del robot se toma positivo hacia arriba.
        """
        points_mm = np.asarray(points_mm, dtype=float)
        scale = self._draw_scale()
        dx = points_mm[:, 0] - (self.p.roi_x_mm + self.p.roi_w_mm / 2.0)
        dy = -(points_mm[:, 1] - (self.p.roi_y_mm + self.p.roi_h_mm / 2.0))
        return np.column_stack((
            self.p.draw_center_x + dx * scale,
            self.p.draw_center_y + dy * scale,
        ))

    def _log_pipeline_stats(self, splines, data, graph_info) -> None:
        kinds = [comp["mark_kind"] for comp in data["mark_components"]]
        self.get_logger().info(
            f"Marcas detectadas: puntos={kinds.count('dot_i_j')}, "
            f"diéresis={kinds.count('diaeresis_dot')}, "
            f"tildes/acentos={kinds.count('accent_or_tilde')}"
        )
        self.get_logger().info(
            f"Grafo: extremos={len(graph_info['endpoints'])} | "
            f"bifurcaciones={len(graph_info['junctions'])}"
        )
        self.get_logger().info(
            f"Caminos: esqueleto={len(data['raw_skeleton_paths'])}, "
            f"marcas={len(data['mark_paths'])}, "
            f"ordenados={len(data['ordered_paths'])}, "
            f"finales={len(data['final_paths'])}"
        )
        self.get_logger().info(
            f"Splines generados: {len(splines)} | retrocesos locales/globales: "
            f"{sum(s['retrace_count'] for s in splines)}/"
            f"{sum(s['global_retrace_count'] for s in splines)}"
        )

    # ------------------------------------------------------------------
    # Construcción de mensajes
    # ------------------------------------------------------------------

    def _build_messages(self, trajectories_xy: list):
        """Concatena las trayectorias; flags marca con 1 el inicio de cada trazo."""
        all_xy = np.concatenate(trajectories_xy, axis=0)
        flags = np.zeros(len(all_xy), dtype=np.int8)
        flags[np.cumsum([0] + [len(t) for t in trajectories_xy[:-1]])] = 1
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
    node = LetterTrajectoryPublisher()
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
