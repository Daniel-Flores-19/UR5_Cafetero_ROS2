#!/usr/bin/env python3
"""
letter_trajectory.py — Publicador ROS2 de trayectorias de letras (pipeline V7 A4).

Flujo:
    texto -> imagen A4 -> binarización/marcas -> esqueleto -> caminos
    -> orden/fusión V7 -> splines -> mm A4 -> XY robot

Publica (al arrancar, o tras /image_drawing/done si esperar_imagen:=true):
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
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, Int8MultiArray, MultiArrayDimension

from ur5_algoritmos.cinematica import TF2xyzquat, fkine_ur5

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

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


# Mapa de la mesa que deja calibration_surface_A4_4x4: de ahí sale el centro
# de la hoja, para que el texto caiga sobre la A4 que se acaba de palpar.
SUPERFICIE_NPZ = "~/pintor_ws/src/calibracion/surface_calibration_A4.npz"

# Parámetros propios del nodo, más los del pipeline que se sobrescriben aquí.
NODE_DEFAULTS = {
    "text": "Universidad de Ingenieria y Tecnologia",
    "font_path": choose_default_font_path(),
    "spline_points": 220,
    # ROI = la hoja A4 entera (vertical, 210 x 297 mm). El margen de 20 mm es el
    # mismo edge_margin de la calibración. La hoja virtual se reduce luego a
    # draw_width x draw_height en el papel real.
    "roi_x_mm": 0.0, "roi_y_mm": 0.0, "roi_w_mm": 210.0, "roi_h_mm": 297.0,
    "roi_margin_mm": 20.0,
    # Mapeo del ROI A4 al plano XY del robot [m]. draw_width y draw_height son
    # la caja LÍMITE del dibujo: se usa una sola escala para X e Y (la mayor que
    # cabe dentro), así que el texto ocupa una de las dos dimensiones y la otra
    # queda holgada, pero sin deformarse. Con 0.105 x 0.1485 (1/4 de A4, o sea
    # un A6: mitad de cada lado) la escala es 0.5, y el margen de 20 mm de la
    # hoja virtual queda en 10 mm reales.
    # Centro del texto. NaN -> se decide con centro_desde_robot:
    #   True  -> el XY donde está el efector AHORA (primer /joint_states + FK):
    #            basta con dejar el plumón sobre el punto donde se quiere el texto.
    #   False -> el centro de la hoja calibrada (superficie_npz).
    "draw_center_x": float("nan"),
    "draw_center_y": float("nan"),
    "centro_desde_robot": True,
    # Desplazamiento [m] del centro respecto a la posición del robot (sólo con
    # centro_desde_robot). -0.0994 en Y deja la palabra 30 mm debajo del dibujo
    # de painting_trajectory lanzado desde la misma posición: el dibujo baja
    # 53.3 mm bajo el robot, +30 mm de espacio, +16.1 mm hasta el centro de la
    # palabra. Si cambia la imagen, el espacio o el tamaño del texto, recalcular.
    "offset_x": 0.0,
    "offset_y": -0.15,
    "draw_width": 0.105,
    "draw_height": 0.1485,
    "superficie_npz": SUPERFICIE_NPZ,
    # True -> espera /image_drawing/done (encadenado tras img_trajectory).
    # False -> genera y publica apenas arranca.
    "esperar_imagen": False,
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
        self._pending_start = False

        explicit = np.isfinite(self.p.draw_center_x) and np.isfinite(self.p.draw_center_y)
        self._center_ready = explicit
        if explicit:
            self.get_logger().info(
                f"Centro de dibujo fijado por parámetro: "
                f"X={self.p.draw_center_x:.4f} Y={self.p.draw_center_y:.4f}")
        elif self.p.centro_desde_robot:
            self._js_sub = self.create_subscription(
                JointState, "/joint_states", self._on_joint_state, 10)
            self.get_logger().info("Esperando /joint_states para tomar la posición actual como centro...")
        else:
            self._center_from_calibration()

        if self.p.esperar_imagen:
            self.create_subscription(Bool, "image_drawing/done", self._on_image_done, 10)
            self.get_logger().info("Esperando señal de fin de dibujo de imagen...")
        else:
            # Un segundo de gracia para que move_draw_sub descubra el publicador.
            self._start_timer = self.create_timer(1.0, self._start_now)

    def _start_now(self) -> None:
        self._start_timer.cancel()
        self._on_image_done(Bool(data=True))

    def _on_joint_state(self, msg: JointState) -> None:
        """Toma UNA vez el XY actual del efector como centro del texto."""
        if self._center_ready:
            return
        try:
            q = np.array([msg.position[msg.name.index(j)] for j in JOINT_NAMES], dtype=float)
        except ValueError:
            return
        x = np.asarray(TF2xyzquat(fkine_ur5(q)), dtype=float)
        self.p.draw_center_x = float(x[0]) + float(self.p.offset_x)
        self.p.draw_center_y = float(x[1]) + float(self.p.offset_y)
        self._center_ready = True
        self.destroy_subscription(self._js_sub)
        self.get_logger().info(
            f"Posición actual del robot: X={x[0]:.4f} Y={x[1]:.4f} (Z={x[2]:.4f}) | "
            f"offset ({self.p.offset_x * 1000:+.1f}, {self.p.offset_y * 1000:+.1f}) mm -> "
            f"centro de dibujo X={self.p.draw_center_x:.4f} Y={self.p.draw_center_y:.4f}")
        if self._pending_start:
            self._on_image_done(Bool(data=True))

    def _center_from_calibration(self) -> None:
        """Centro de la hoja calibrada, leído de superficie_npz."""
        npz = os.path.expanduser(self.p.superficie_npz)
        try:
            cx, cy = (float(v) for v in np.load(npz)["centro"])
        except (OSError, KeyError, ValueError) as error:
            raise RuntimeError(
                f"draw_center_x/y sin definir y no se pudo leer el centro de {npz}: {error}. "
                "Corre la calibración o pasa -p draw_center_x:=... -p draw_center_y:=...") from error
        self.p.draw_center_x, self.p.draw_center_y = cx, cy
        self._center_ready = True
        self.get_logger().info(
            f"Centro de dibujo tomado de la calibración ({npz}): "
            f"X={self.p.draw_center_x:.4f} Y={self.p.draw_center_y:.4f}")

    # ------------------------------------------------------------------
    # Arranque y publicación periódica
    # ------------------------------------------------------------------

    def _on_image_done(self, msg: Bool) -> None:
        if not msg.data or self._started:
            return
        if not self._center_ready:
            # Se arranca en cuanto llegue la primera lectura de /joint_states.
            self._pending_start = True
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
