#!/usr/bin/env python3
"""
letter_trajectory_publisher.py

Publishes the full trajectory as two arrays, repeated PUBLISH_REPEATS times
at PUBLISH_HZ so late-joining subscribers don't miss it.

    /letter_trajectory/xy      (Float32MultiArray)  [x0,y0, x1,y1, ..., xN,yN]
    /letter_trajectory/flags   (Int8MultiArray)      [1,0,0,...,1,0,0,...]
                                                       ^ new segment
"""
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int8MultiArray, MultiArrayDimension
import matplotlib.pyplot as plt

from ur5_algoritmos.trayectorias.imagen_functions import (
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
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "Tipografia", "Playwrite_CU", "PlaywriteCU-VariableFont_wght.ttf"),
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


# ---------------------------------------------------------------------------
# Hoja de papel
#
# A4 en VERTICAL sobre la mesa: el lado corto (210 mm) va sobre X del robot y
# el largo (297 mm) sobre Y. Con 20 mm de margen el área de dibujo es
# 170 x 257 mm, que es EXACTAMENTE la zona que palpa calibration_surface_A4_3x3
# con sus valores por defecto, así que el modelo de la mesa nunca extrapola.
#
# El centro y el área salen del .npz de la calibración (ver area_calibrada más
# abajo); los valores de aquí son solo el respaldo previo a calibrar.
# ---------------------------------------------------------------------------
PAPER_X      = 0.210   # [m] lado de la A4 sobre X del robot
PAPER_Y      = 0.297   # [m] lado de la A4 sobre Y del robot
PAPER_MARGIN = 0.020   # [m] margen libre en cada borde

# Respaldo si todavía no se ha calibrado nunca. En cuanto existe el .npz manda
# la calibración: es la única fuente de verdad sobre dónde está la hoja.
DRAW_CENTER = np.array([-0.5824, 0.0653])
DRAW_WIDTH  = PAPER_X - 2 * PAPER_MARGIN   # 0.170 m
DRAW_HEIGHT = PAPER_Y - 2 * PAPER_MARGIN   # 0.257 m

SUPERFICIE_NPZ = '~/pintorV2_ws/calibracion/surface_calibration_A4.npz'


def area_calibrada(ruta=SUPERFICIE_NPZ):
    """Área de dibujo tal como la dejó la calibración de superficie.

    El .npz guarda el centro palpado, el tamaño de hoja y el margen, así que el
    área sale de ahí en vez de estar copiada a mano en este archivo: si la hoja
    se mueve, basta con recalibrar y el dibujo la sigue. Devolver exactamente la
    zona palpada es además lo que garantiza que el modelo z(x, y) no extrapole.

    Devuelve (centro, ancho, alto) o None si no hay calibración utilizable.
    """
    ruta_abs = os.path.expanduser(ruta)

    if not os.path.exists(ruta_abs):
        return None

    try:
        datos = np.load(ruta_abs)
        centro = np.asarray(datos['centro'], dtype=float)
        paper = np.asarray(datos['paper'], dtype=float)
        margen = float(datos['edge_margin'])
    except (OSError, KeyError, ValueError):
        return None

    return centro, float(paper[0]) - 2 * margen, float(paper[1]) - 2 * margen


_area = area_calibrada()

if _area is not None:
    DRAW_CENTER, DRAW_WIDTH, DRAW_HEIGHT = _area
    print(f"[imagen_trajectory] Área de dibujo desde la calibración: "
          f"centro=({DRAW_CENTER[0]:.4f}, {DRAW_CENTER[1]:.4f}) "
          f"{DRAW_WIDTH*1000:.0f}x{DRAW_HEIGHT*1000:.0f} mm")
else:
    print(f"[imagen_trajectory] Sin calibración en {SUPERFICIE_NPZ}: "
          f"usando el área por defecto "
          f"centro=({DRAW_CENTER[0]:.4f}, {DRAW_CENTER[1]:.4f}). "
          f"Corre calibration_surface_A4_3x3 antes de dibujar.")

# Gira el dibujo 90° sobre la hoja: el ancho de la imagen pasa a recorrer el
# lado LARGO del papel. Con imágenes apaisadas permite dibujarlas mucho más
# grandes sobre una A4 vertical.
DRAW_ROTATE_90 = True

PUBLISH_REPEATS = 10     # how many times to send both arrays
PUBLISH_HZ      = 1.0   # rate between repetitions


# ===========================================================================
# Utilidad
# ===========================================================================

def pixel_to_robot_xy(px, py, img_size):
    """
    Píxel de la imagen -> XY del robot sobre la hoja.

    La escala es la MISMA en los dos ejes, la mayor que deja la imagen entera
    dentro del área de dibujo: así el trazo conserva su forma. Antes se
    normalizaba cada eje por separado, lo que estiraba el dibujo hasta llenar
    el rectángulo sin importar la forma de la imagen.

    Con DRAW_ROTATE_90 el dibujo se gira un cuarto de vuelta: el ancho de la
    imagen recorre el lado largo de la hoja y el alto el corto. Es un giro, no
    un espejo, así que el dibujo no sale invertido.

    La imagen queda centrada en DRAW_CENTER.
    """
    alto, ancho = img_size[0], img_size[1]

    u = px - ancho / 2.0     # +u = hacia la derecha de la imagen
    v = py - alto / 2.0      # +v = hacia abajo de la imagen

    if DRAW_ROTATE_90:
        # El alto de la imagen cae sobre X del robot y el ancho sobre Y.
        escala = min(DRAW_WIDTH / alto, DRAW_HEIGHT / ancho)
        return np.array([
            DRAW_CENTER[0] + v * escala,
            DRAW_CENTER[1] + u * escala,
        ])

    escala = min(DRAW_WIDTH / ancho, DRAW_HEIGHT / alto)   # [m] por píxel

    return np.array([
        DRAW_CENTER[0] + u * escala,
        DRAW_CENTER[1] - v * escala,
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
        self.done = False
        #self._plot_published_data()
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
        font_name, font_path = FONTS[FONT_KEY]
        self.get_logger().info(f"Texto: '{LETTER}' | Fuente: {font_name}")

        # Processing an image
        image = cv2.imread('/home/mito/Downloads/kirby.png')    
        img_resized = ur5_resize_paper(image)
        img_canny = auto_canny_limits(img_resized)
        

        img_canny_f = filter_canny_by_density(
            img_canny,
            window_size=33,
            density_threshold=0.1,
            suppression_mode="thin",
            )
        # Se esqueletiza la versión filtrada por densidad. Con los parámetros
        # actuales sale igual que img_canny (Canny ya da bordes de 1 px y el
        # modo "thin" no tiene nada que adelgazar), pero así afinar el filtro
        # surte efecto sin tener que tocar esta línea.
        _, _, skeleton, _ = process_image(img_canny_f)
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
            np.array([pixel_to_robot_xy(px, py, img_canny_f.shape) for px, py in traj])
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
