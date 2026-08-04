#!/usr/bin/env python3
"""
letter_trajectory_functions.py  —  V8 (escritura humana)

Funciones para generar trayectorias a partir de letras o palabras.

Flujo:
    1. Generar imagen de la palabra con PIL.
    2. Binarizar la imagen.
    3. Obtener el esqueleto de la palabra.
    4. Convertir el esqueleto en un grafo.
    5. Detectar extremos y bifurcaciones.
    6. Extraer segmentos ordenados.
    7. Fusionar con lógica de escritura humana.
    8. Suavizar cada segmento usando splines.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize


# ===========================================================================
# 1. Generación de imagen
# ===========================================================================

def get_next_filename(output_dir: str, text: str, font_name: str) -> str:
    """Genera un nombre de archivo incremental para evitar sobrescribir imágenes."""
    safe_text = str(text).replace(" ", "_")
    i = 1
    while True:
        file_name = f"{safe_text}_{font_name}_{i}.png"
        file_path = os.path.join(output_dir, file_name)
        if not os.path.exists(file_path):
            return file_path
        i += 1


def plot_letter_pil(
    text: str,
    font_path: str,
    output_dir: str = "letras_png",
    size: int = 800,
    max_font_size: int = 420,
    margin: int = 40,
) -> str:
    """
    Genera una imagen PNG de una letra o palabra con la fuente indicada.

    Retorna la ruta del archivo guardado.
    """
    os.makedirs(output_dir, exist_ok=True)

    font_name = os.path.basename(font_path).split(".")[0]
    file_path = get_next_filename(output_dir, text, font_name)

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    max_w = size - 2 * margin
    max_h = size - 2 * margin

    font_size = max_font_size
    while font_size > 10:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= max_w and text_h <= max_h:
            break
        font_size -= 5

    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (size - text_w) / 2 - bbox[0]
    y = (size - text_h) / 2 - bbox[1]

    draw.text((x, y), text, fill="black", font=font)
    img.save(file_path)

    print(f"Imagen guardada en: {file_path}")
    print(f"Tamaño de fuente usado: {font_size}")

    return file_path


# ===========================================================================
# 2. Procesamiento de imagen y esqueleto
# ===========================================================================

# ===========================================================================
# 2. Procesamiento de imagen y esqueleto
# ===========================================================================

def auto_canny_limits(image, sigma=0.33):

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, lower, upper)

    return edges


def ur5_resize_paper(image):
    max_height = 2000
    max_width = 3000

    img_height = image.shape[0]
    img_width = image.shape[1]

    if (img_height > 0.8*max_height) or (img_width > 0.8 * max_width):
        target_size = [int(max_width * 0.8), int(max_height * 0.8)]
        resized_img = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        return resized_img

    else:
        return image
        
def process_image(file_path):
    """
    Procesa la imagen de una palabra y obtiene su esqueleto.

    Retorna:
        img_skeleton  : imagen original con el esqueleto pintado en rojo.
        binary_clean  : imagen binaria procesada.
        skeleton_uint8: imagen del esqueleto en 0/255.
        gray          : imagen en escala de grises.
    """
    threshold_value = 200
    close_kernel_size = 5
    close_iterations = 1
    erosion_kernel_size = 3
    erosion_iterations = 0
    skeleton_thickness = 3

    if type(file_path) == str:
        img = cv2.imread(file_path)
    else:
        img = file_path
    if img is None:
        print("Error: no se pudo cargar la imagen")
        return None, None, None, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)

    if erosion_iterations > 0:
        erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        binary_clean = cv2.erode(binary_clean, erosion_kernel, iterations=erosion_iterations)

    binary_bool = binary_clean > 0
    skeleton = skeletonize(binary_bool)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    skeleton_view = cv2.dilate(
        skeleton_uint8,
        np.ones((skeleton_thickness, skeleton_thickness), np.uint8),
        iterations=1,
    )

    img_skeleton = img.copy()
    y_coords, x_coords = np.where(skeleton_view > 0)
    img_skeleton[y_coords, x_coords] = (0, 0, 255)

    return img_skeleton, binary_clean, skeleton_uint8, gray


# ===========================================================================
# 3. Esqueleto → puntos
# ===========================================================================

def skeleton_to_points(skeleton: np.ndarray, step: int = 3):
    """Convierte una imagen de esqueleto en una nube de puntos (N, 2) [x, y]."""
    y_coords, x_coords = np.where(skeleton > 0)
    if len(x_coords) == 0:
        return None
    points = np.column_stack((x_coords, y_coords))
    return points[::step]


def draw_points_on_image(img: np.ndarray, points, radius: int = 2) -> np.ndarray:
    """Dibuja los puntos extraídos encima de la imagen original."""
    img_points = img.copy()
    if points is None:
        return img_points
    for p in points:
        cv2.circle(img_points, (int(p[0]), int(p[1])), radius, (255, 0, 0), -1)
    return img_points


# ===========================================================================
# 4. Análisis de puntos clave del esqueleto
# ===========================================================================

def analyze_skeleton_points(skeleton: np.ndarray):
    """
    Detecta extremos, puntos normales y bifurcaciones en un esqueleto.

    Retorna:
        endpoints     : array de extremos (1 vecino).
        junctions     : array de bifurcaciones (≥3 vecinos).
        normal_points : array de puntos normales (2 vecinos).
    """
    skeleton_bin = (skeleton > 0).astype(np.uint8)
    endpoints, junctions, normal_points = [], [], []
    h, w = skeleton_bin.shape

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton_bin[y, x] == 0:
                continue
            neighborhood = skeleton_bin[y - 1 : y + 2, x - 1 : x + 2]
            num_neighbors = int(np.sum(neighborhood) - 1)
            if num_neighbors == 1:
                endpoints.append([x, y])
            elif num_neighbors == 2:
                normal_points.append([x, y])
            elif num_neighbors >= 3:
                junctions.append([x, y])

    return np.array(endpoints), np.array(junctions), np.array(normal_points)


def draw_keypoints_on_skeleton(
    skeleton: np.ndarray,
    endpoints: np.ndarray,
    junctions: np.ndarray,
) -> np.ndarray:
    """Dibuja extremos en verde y bifurcaciones en rojo sobre el esqueleto."""
    img_debug = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    for x, y in endpoints:
        cv2.circle(img_debug, (int(x), int(y)), 5, (0, 255, 0), -1)
    for x, y in junctions:
        cv2.circle(img_debug, (int(x), int(y)), 5, (0, 0, 255), -1)
    return img_debug


# ===========================================================================
# 5. Esqueleto → segmentos ordenados
# ===========================================================================

def get_neighbors(point: tuple, skeleton_bin: np.ndarray) -> list:
    """Retorna vecinos 8-conectados de un punto (x, y) del esqueleto."""
    x, y = point
    neighbors = []
    h, w = skeleton_bin.shape
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and skeleton_bin[ny, nx] > 0:
                neighbors.append((nx, ny))
    return neighbors


def _skeleton_degree_image(skeleton_bin: np.ndarray) -> np.ndarray:
    """Calcula el número de vecinos 8-conectados de cada píxel del esqueleto."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton_bin, -1, kernel)
    degree = neighbor_count - skeleton_bin
    degree[skeleton_bin == 0] = 0
    return degree


def _build_keypoint_nodes(skeleton_bin: np.ndarray, junction_dilate: int = 1):
    """
    Construye nodos del grafo agrupando nubes de bifurcación en un único nodo.

    Retorna:
        node_map   : dict {(x,y) -> node_id}
        node_points: dict {node_id -> (cx, cy)}
        node_type  : dict {node_id -> "junction"|"endpoint"}
    """
    degree = _skeleton_degree_image(skeleton_bin)

    endpoint_mask = ((skeleton_bin > 0) & (degree == 1)).astype(np.uint8)
    junction_mask = ((skeleton_bin > 0) & (degree >= 3)).astype(np.uint8)

    if junction_dilate > 0:
        kernel = np.ones((3, 3), np.uint8)
        junction_mask_grouped = cv2.dilate(junction_mask, kernel, iterations=junction_dilate)
        junction_mask_grouped = (junction_mask_grouped & skeleton_bin).astype(np.uint8)
    else:
        junction_mask_grouped = junction_mask

    num_labels, labels = cv2.connectedComponents(junction_mask_grouped, connectivity=8)

    node_map, node_points, node_type = {}, {}, {}
    node_id = 0

    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        pts = [(int(x), int(y)) for x, y in zip(xs, ys)]
        cx, cy = int(round(np.mean(xs))), int(round(np.mean(ys)))
        node_points[node_id] = (cx, cy)
        node_type[node_id] = "junction"
        for p in pts:
            node_map[p] = node_id
        node_id += 1

    ys, xs = np.where(endpoint_mask > 0)
    for x, y in zip(xs, ys):
        p = (int(x), int(y))
        if p in node_map:
            continue
        node_points[node_id] = p
        node_type[node_id] = "endpoint"
        node_map[p] = node_id
        node_id += 1

    return node_map, node_points, node_type


def skeleton_to_ordered_segments(
    skeleton: np.ndarray,
    min_length: int = 6,
    junction_dilate: int = 1,
    keep_cycles: bool = True,
) -> list:
    """
    Convierte el esqueleto en segmentos ordenados.

    Evita:
    - Demasiados trazos por nubes de bifurcación.
    - Pérdida de trazos al reducir agresivamente.
    """
    skeleton_bin = (skeleton > 0).astype(np.uint8)
    node_map, node_points, _ = _build_keypoint_nodes(skeleton_bin, junction_dilate=junction_dilate)

    all_pixels_yx = np.column_stack(np.where(skeleton_bin > 0))
    all_points = set((int(x), int(y)) for y, x in all_pixels_yx)

    visited_edges: set = set()
    segments: list = []

    def edge_key(p1, p2):
        return tuple(sorted([p1, p2]))

    def point_to_node(point):
        return node_map.get(point, None)

    # --- Caminos que nacen desde nodos ---
    for start_pixel, start_node in list(node_map.items()):
        for neighbor in get_neighbors(start_pixel, skeleton_bin):
            ekey = edge_key(start_pixel, neighbor)
            if ekey in visited_edges:
                continue

            segment = [node_points[start_node]]
            prev = start_pixel
            current = neighbor
            visited_edges.add(ekey)

            while True:
                current_node = point_to_node(current)

                if current_node is not None and current_node != start_node:
                    segment.append(node_points[current_node])
                    break

                if current_node == start_node:
                    next_candidates = [p for p in get_neighbors(current, skeleton_bin) if p != prev]
                else:
                    segment.append(current)
                    next_candidates = [p for p in get_neighbors(current, skeleton_bin) if p != prev]

                if not next_candidates:
                    break

                next_point = None
                for candidate in next_candidates:
                    if edge_key(current, candidate) not in visited_edges:
                        next_point = candidate
                        break

                if next_point is None:
                    break

                visited_edges.add(edge_key(current, next_point))
                prev = current
                current = next_point

            clean_segment = []
            for p in segment:
                if not clean_segment or p != clean_segment[-1]:
                    clean_segment.append(p)

            if len(clean_segment) >= min_length:
                segments.append(np.array(clean_segment, dtype=float))

    # --- Ciclos cerrados sin extremos ni bifurcaciones ---
    if keep_cycles:
        used_points: set = set()
        for seg in segments:
            for x, y in seg:
                used_points.add((int(round(x)), int(round(y))))
        used_points |= set(node_map.keys())

        remaining_points = all_points - used_points

        while remaining_points:
            start = next(iter(remaining_points))
            segment = [start]
            remaining_points.discard(start)

            prev = None
            current = start

            while True:
                neighbors = get_neighbors(current, skeleton_bin)
                if prev is not None:
                    neighbors = [p for p in neighbors if p != prev]

                next_point = next((p for p in neighbors if p in remaining_points), None)
                if next_point is None:
                    break

                segment.append(next_point)
                remaining_points.discard(next_point)
                prev = current
                current = next_point

                if len(segment) > min_length:
                    if np.linalg.norm(np.array(current) - np.array(start)) <= 2:
                        break

            if len(segment) >= min_length:
                segments.append(np.array(segment, dtype=float))

    return segments


# ===========================================================================
# 6. Utilidades geométricas compartidas
# ===========================================================================

def _remove_consecutive_duplicates(segment: np.ndarray) -> np.ndarray:
    clean = []
    for p in segment:
        if not clean or tuple(p) != tuple(clean[-1]):
            clean.append(p)
    return np.array(clean, dtype=float)


def segment_length(segment: np.ndarray) -> float:
    """Calcula la longitud aproximada de un segmento."""
    if len(segment) < 2:
        return 0.0
    diffs = np.diff(segment, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def is_closed_segment(segment: np.ndarray, distance_threshold: float = 5.0) -> bool:
    """Determina si un segmento se comporta como un ciclo cerrado."""
    if len(segment) < 4:
        return False
    return bool(np.linalg.norm(segment[0] - segment[-1]) <= distance_threshold)


def get_segment_direction(segment: np.ndarray, at_start: bool = True, n: int = 8, eps: float = 1e-9):
    """Calcula una dirección local robusta del segmento."""
    segment = np.asarray(segment, dtype=float)
    segment = _remove_consecutive_duplicates(segment)
    if len(segment) < 2:
        return None

    n = min(n, len(segment) - 1)

    if at_start:
        p0 = segment[0]
        for k in range(1, n + 1):
            d = segment[k] - p0
            norm = np.linalg.norm(d)
            if norm > eps:
                return d / norm
    else:
        p0 = segment[-1]
        for k in range(2, n + 2):
            d = p0 - segment[-k]
            norm = np.linalg.norm(d)
            if norm > eps:
                return d / norm
    return None


def safe_neg(v):
    """Devuelve -v si v no es None."""
    return None if v is None else -v


def angle_between_vectors(v1, v2) -> float:
    """Calcula el ángulo en grados entre dos vectores."""
    if v1 is None or v2 is None:
        return 180.0
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def _angle_between_vectors_soft(v1, v2, default_angle: float = 90.0) -> float:
    if v1 is None or v2 is None:
        return default_angle
    return angle_between_vectors(v1, v2)


def _rotate_closed_segment(segment: np.ndarray, start_index: int) -> np.ndarray:
    """Reordena un ciclo cerrado para que empiece en start_index."""
    segment = _remove_consecutive_duplicates(np.asarray(segment, dtype=float))
    if len(segment) < 4:
        return segment

    if np.linalg.norm(segment[0] - segment[-1]) <= 5:
        base = segment[:-1]
    else:
        base = segment

    start_index = int(np.clip(start_index, 0, len(base) - 1))
    rotated = np.vstack((base[start_index:], base[: start_index + 1]))
    return _remove_consecutive_duplicates(rotated)


def _open_cycles_near_open_segments(
    closed_segments: list,
    open_segments: list,
    connection_distance: float = 18.0,
):
    """Convierte ciclos cerrados en trayectorias abiertas cuando están cerca de otro trazo."""
    opened_cycles, isolated_cycles = [], []

    open_endpoints = []
    for seg in open_segments:
        if len(seg) >= 2:
            open_endpoints.extend([seg[0], seg[-1]])

    if not open_endpoints:
        return closed_segments, opened_cycles

    open_endpoints = np.asarray(open_endpoints, dtype=float)

    for cyc in closed_segments:
        cyc = _remove_consecutive_duplicates(np.asarray(cyc, dtype=float))
        if len(cyc) < 4:
            isolated_cycles.append(cyc)
            continue

        best_i, best_d = None, np.inf
        for i, p in enumerate(cyc):
            d = float(np.min(np.linalg.norm(open_endpoints - p, axis=1)))
            if d < best_d:
                best_d, best_i = d, i

        if best_d <= connection_distance:
            opened_cycles.append(_rotate_closed_segment(cyc, best_i))
        else:
            isolated_cycles.append(cyc)

    return isolated_cycles, opened_cycles


def _join_segments_with_bridge(
    current: np.ndarray,
    candidate: np.ndarray,
    mode: str,
    bridge_points: int = 5,
) -> np.ndarray:
    """Une dos segmentos con la orientación correcta, insertando puntos intermedios si hay hueco."""
    if mode == "end_start":
        left, right = current, candidate
    elif mode == "end_end":
        left, right = current, candidate[::-1]
    elif mode == "start_end":
        left, right = candidate, current
    elif mode == "start_start":
        left, right = candidate[::-1], current
    else:
        raise ValueError(f"Modo de unión no reconocido: {mode}")

    p0, p1 = left[-1], right[0]
    d = np.linalg.norm(p1 - p0)

    if d > 1e-9 and bridge_points > 0:
        bridge = np.linspace(p0, p1, bridge_points + 2)[1:-1]
        joined = np.vstack((left, bridge, right))
    else:
        joined = np.vstack((left, right))

    return _remove_consecutive_duplicates(joined)


# ===========================================================================
# 7. Fusión de segmentos — versión geométrica (V7, conservada)
# ===========================================================================

def merge_connected_segments(
    segments: list,
    max_endpoint_distance: float = 10.0,
    max_angle: float = 70.0,
    min_branch_length: float = 0.0,
    keep_closed_cycles: bool = True,
    protect_short_isolated: bool = True,
) -> list:
    """
    Une segmentos por proximidad y continuidad angular (versión simple).

    Para un ajuste rápido sin volver a procesar la imagen.
    """
    closed_segments, open_segments = [], []

    for seg in segments:
        seg = _remove_consecutive_duplicates(np.asarray(seg, dtype=float))
        if len(seg) < 2 or segment_length(seg) < 1e-6:
            continue

        if keep_closed_cycles and is_closed_segment(seg):
            closed_segments.append(seg)
            continue

        if min_branch_length > 0 and segment_length(seg) < min_branch_length:
            if protect_short_isolated:
                open_segments.append(seg)
        else:
            open_segments.append(seg)

    segments = open_segments
    used = [False] * len(segments)
    merged_segments = []

    for i in range(len(segments)):
        if used[i]:
            continue

        current = segments[i].copy()
        used[i] = True
        changed = True

        while changed:
            changed = False
            current_start = current[0]
            current_end = current[-1]
            current_dir_start = get_segment_direction(current, at_start=True)
            current_dir_end = get_segment_direction(current, at_start=False)

            best_j, best_mode, best_score = None, None, np.inf

            for j in range(len(segments)):
                if used[j]:
                    continue

                candidate = segments[j]
                cases = [
                    (np.linalg.norm(current_end - candidate[0]),
                     angle_between_vectors(current_dir_end, get_segment_direction(candidate, True)),
                     "end_start"),
                    (np.linalg.norm(current_end - candidate[-1]),
                     angle_between_vectors(current_dir_end, safe_neg(get_segment_direction(candidate, False))),
                     "end_end"),
                    (np.linalg.norm(current_start - candidate[-1]),
                     angle_between_vectors(safe_neg(current_dir_start), get_segment_direction(candidate, False)),
                     "start_end"),
                    (np.linalg.norm(current_start - candidate[0]),
                     angle_between_vectors(safe_neg(current_dir_start), safe_neg(get_segment_direction(candidate, True))),
                     "start_start"),
                ]

                for d, a, mode in cases:
                    if d <= max_endpoint_distance and a <= max_angle:
                        score = 3.0 * a + d
                        if score < best_score:
                            best_score, best_j, best_mode = score, j, mode

            if best_j is not None:
                candidate = segments[best_j]
                if best_mode == "end_start":
                    current = np.vstack((current, candidate))
                elif best_mode == "end_end":
                    current = np.vstack((current, candidate[::-1]))
                elif best_mode == "start_end":
                    current = np.vstack((candidate, current))
                elif best_mode == "start_start":
                    current = np.vstack((candidate[::-1], current))
                current = _remove_consecutive_duplicates(current)
                used[best_j] = True
                changed = True

        merged_segments.append(current)

    return closed_segments + merged_segments


def merge_segments_by_best_continuation(
    segments: list,
    max_endpoint_distance: float = 22.0,
    max_angle: float = 105.0,
    bridge_points: int = 5,
    min_branch_length: float = 0.0,
    keep_closed_cycles: bool = True,
    open_closed_cycles: bool = True,
    cycle_connection_distance: float = 20.0,
    protect_short_isolated: bool = True,
    distance_weight: float = 1.0,
    angle_weight: float = 0.85,
    length_bonus: float = 0.10,
    close_distance_angle_relax: float = 6.0,
    relaxed_max_angle: float = 145.0,
) -> list:
    """
    Fusiona segmentos siguiendo la continuación geométrica más natural del trazo.

    En cada iteración evalúa cuatro casos de unión y elige el de menor costo:
        costo = distancia_norm + peso_angular * angulo_norm - bono_longitud
    """
    closed_segments, open_segments = [], []

    for seg in segments:
        seg = _remove_consecutive_duplicates(np.asarray(seg, dtype=float))
        if len(seg) < 2 or segment_length(seg) < 1e-6:
            continue

        if keep_closed_cycles and is_closed_segment(seg):
            closed_segments.append(seg)
            continue

        if min_branch_length > 0 and segment_length(seg) < min_branch_length:
            if protect_short_isolated:
                open_segments.append(seg)
        else:
            open_segments.append(seg)

    isolated_cycles = closed_segments

    if keep_closed_cycles and open_closed_cycles:
        isolated_cycles, opened_cycles = _open_cycles_near_open_segments(
            closed_segments, open_segments, connection_distance=cycle_connection_distance
        )
        open_segments = open_segments + opened_cycles

    segments = open_segments
    if not segments:
        return isolated_cycles

    lengths = np.array([max(segment_length(s), 1e-9) for s in segments], dtype=float)
    max_len = float(np.max(lengths))

    used = [False] * len(segments)
    merged_segments = []
    order = np.argsort(-lengths)

    for i in order:
        if used[i]:
            continue

        current = segments[i].copy()
        used[i] = True
        changed = True

        while changed:
            changed = False
            current_start = current[0]
            current_end = current[-1]
            current_dir_start = get_segment_direction(current, at_start=True)
            current_dir_end = get_segment_direction(current, at_start=False)

            best_j, best_mode, best_score = None, None, np.inf

            for j in range(len(segments)):
                if used[j]:
                    continue

                candidate = segments[j]
                cand_start = candidate[0]
                cand_end = candidate[-1]
                cand_dir_start = get_segment_direction(candidate, at_start=True)
                cand_dir_end = get_segment_direction(candidate, at_start=False)

                cases = [
                    (np.linalg.norm(current_end - cand_start),
                     _angle_between_vectors_soft(current_dir_end, cand_dir_start),
                     "end_start"),
                    (np.linalg.norm(current_end - cand_end),
                     _angle_between_vectors_soft(current_dir_end, safe_neg(cand_dir_end)),
                     "end_end"),
                    (np.linalg.norm(current_start - cand_end),
                     _angle_between_vectors_soft(safe_neg(current_dir_start), cand_dir_end),
                     "start_end"),
                    (np.linalg.norm(current_start - cand_start),
                     _angle_between_vectors_soft(safe_neg(current_dir_start), safe_neg(cand_dir_start)),
                     "start_start"),
                ]

                for d, a, mode in cases:
                    local_max_angle = relaxed_max_angle if d <= close_distance_angle_relax else max_angle
                    if d <= max_endpoint_distance and a <= local_max_angle:
                        d_norm = d / max(max_endpoint_distance, 1e-9)
                        a_norm = a / max(local_max_angle, 1e-9)
                        l_norm = segment_length(candidate) / max_len
                        score = distance_weight * d_norm + angle_weight * a_norm - length_bonus * l_norm
                        if score < best_score:
                            best_score, best_j, best_mode = score, j, mode

            if best_j is not None:
                current = _join_segments_with_bridge(current, segments[best_j], best_mode, bridge_points)
                used[best_j] = True
                changed = True

        merged_segments.append(current)

    return isolated_cycles + merged_segments


# ===========================================================================
# 8. Fusión con lógica de escritura humana  — NUEVO en V8
# ===========================================================================

def _endpoint_cases(current: np.ndarray, candidate: np.ndarray) -> list:
    """
    Evalúa las 4 formas posibles de orientar dos segmentos.

    Retorna lista de tuplas: (distancia, ángulo, modo).
    """
    current_start  = current[0]
    current_end    = current[-1]
    candidate_start = candidate[0]
    candidate_end   = candidate[-1]

    current_dir_start   = get_segment_direction(current,   at_start=True)
    current_dir_end     = get_segment_direction(current,   at_start=False)
    candidate_dir_start = get_segment_direction(candidate, at_start=True)
    candidate_dir_end   = get_segment_direction(candidate, at_start=False)

    return [
        (np.linalg.norm(current_end   - candidate_start),
         _angle_between_vectors_soft(current_dir_end,          candidate_dir_start),
         "end_start"),
        (np.linalg.norm(current_end   - candidate_end),
         _angle_between_vectors_soft(current_dir_end,          safe_neg(candidate_dir_end)),
         "end_end"),
        (np.linalg.norm(current_start - candidate_end),
         _angle_between_vectors_soft(safe_neg(current_dir_start), candidate_dir_end),
         "start_end"),
        (np.linalg.norm(current_start - candidate_start),
         _angle_between_vectors_soft(safe_neg(current_dir_start), safe_neg(candidate_dir_start)),
         "start_start"),
    ]


def _segment_bbox(segment: np.ndarray) -> dict:
    """Calcula la caja delimitadora de un segmento."""
    segment = np.asarray(segment, dtype=float)
    return {
        "xmin": float(np.min(segment[:, 0])),
        "xmax": float(np.max(segment[:, 0])),
        "ymin": float(np.min(segment[:, 1])),
        "ymax": float(np.max(segment[:, 1])),
        "cx":   float(np.mean(segment[:, 0])),
        "cy":   float(np.mean(segment[:, 1])),
        "w":    float(np.ptp(segment[:, 0])),
        "h":    float(np.ptp(segment[:, 1])),
    }


def _bbox_horizontal_gap(b1: dict, b2: dict) -> float:
    """Distancia horizontal entre dos cajas. Si se solapan retorna 0."""
    if b1["xmax"] < b2["xmin"]:
        return b2["xmin"] - b1["xmax"]
    if b2["xmax"] < b1["xmin"]:
        return b1["xmin"] - b2["xmax"]
    return 0.0


def _bbox_vertical_overlap_ratio(b1: dict, b2: dict) -> float:
    """Solapamiento vertical normalizado entre dos cajas."""
    top    = max(b1["ymin"], b2["ymin"])
    bottom = min(b1["ymax"], b2["ymax"])
    overlap = max(0.0, bottom - top)
    denom   = max(1.0, min(b1["h"], b2["h"]))
    return overlap / denom


def _orient_segment_for_human_start(segment: np.ndarray) -> np.ndarray:
    """
    Orienta un segmento para iniciar de forma más parecida a escritura manual:
    se prefiere comenzar por el extremo más a la izquierda; si hay empate,
    por el extremo superior.
    """
    segment = np.asarray(segment, dtype=float)
    p0 = segment[0]
    p1 = segment[-1]

    # En imágenes, y crece hacia abajo. Menor y = más arriba.
    score0 = p0[0] + 0.15 * p0[1]
    score1 = p1[0] + 0.15 * p1[1]

    if score1 < score0:
        return segment[::-1].copy()
    return segment.copy()


def _candidate_letter_penalty(
    current: np.ndarray,
    candidate: np.ndarray,
    expected_letter_gap: float,
) -> float:
    """
    Penaliza uniones que saltan demasiado horizontalmente o parecen pertenecer
    a letras no vecinas. No prohíbe unir letras cursivas, pero evita saltos raros.
    """
    b1 = _segment_bbox(current)
    b2 = _segment_bbox(candidate)

    hgap    = _bbox_horizontal_gap(b1, b2)
    voverlap = _bbox_vertical_overlap_ratio(b1, b2)

    penalty = 0.0

    if hgap > expected_letter_gap:
        penalty += (hgap / max(expected_letter_gap, 1e-9)) ** 2

    if hgap > 0 and voverlap < 0.15:
        penalty += 0.75

    return penalty


def _choose_next_human_like(
    current: np.ndarray,
    segments: list,
    used: list,
    max_endpoint_distance: float,
    max_angle: float,
    expected_letter_gap: float,
    distance_weight: float,
    angle_weight: float,
    letter_weight: float,
    forward_weight: float,
    length_bonus: float,
    relaxed_close_distance: float,
    relaxed_max_angle: float,
):
    """
    Escoge el siguiente segmento con una función de costo pensada para escritura:
    continuidad local + proximidad + avance izquierda-derecha + no saltar de letra.
    """
    current_bbox   = _segment_bbox(current)
    current_end_x  = current[-1][0]

    lengths  = [max(segment_length(s), 1e-9) for s in segments]
    max_len  = max(lengths) if lengths else 1.0

    best       = None
    best_score = np.inf

    for j, candidate in enumerate(segments):
        if used[j]:
            continue

        cand_bbox = _segment_bbox(candidate)

        for d, a, mode in _endpoint_cases(current, candidate):
            local_angle = relaxed_max_angle if d <= relaxed_close_distance else max_angle

            if d > max_endpoint_distance or a > local_angle:
                continue

            d_norm = d / max(max_endpoint_distance, 1e-9)
            a_norm = a / max(local_angle, 1e-9)
            l_norm = segment_length(candidate) / max_len

            letter_penalty = _candidate_letter_penalty(current, candidate, expected_letter_gap)

            # Penalizar retrocesos fuertes; un pequeño regreso sí se permite.
            if mode in ["end_start", "end_end"]:
                next_x   = candidate[0][0] if mode == "end_start" else candidate[-1][0]
                backward = max(0.0, current_end_x - next_x)
            else:
                backward = max(0.0, current_bbox["xmin"] - cand_bbox["xmax"]) + 8.0

            backward_norm = backward / max(expected_letter_gap, 1e-9)

            score = (
                distance_weight * d_norm
                + angle_weight  * a_norm
                + letter_weight * letter_penalty
                + forward_weight * backward_norm
                - length_bonus  * l_norm
            )

            if score < best_score:
                best_score = score
                best = (j, mode, score)

    return best


def _sort_human_writing_order(segments: list) -> list:
    """
    Ordena trayectorias de izquierda a derecha.

    Los trazos muy pequeños ubicados arriba (puntos, tildes) se mantienen
    después del cuerpo cercano para no mezclarlos con el trazo principal.
    """
    if len(segments) <= 1:
        return segments

    bboxes  = [_segment_bbox(s) for s in segments]
    lengths = np.array([segment_length(s) for s in segments], dtype=float)
    median_len = float(np.median(lengths)) if len(lengths) else 1.0
    median_h   = float(np.median([b["h"] for b in bboxes]))

    decorated = []
    for idx, (seg, box, length) in enumerate(zip(segments, bboxes, lengths)):
        is_small_mark = length < 0.35 * median_len and box["h"] < 0.45 * median_h
        mark_shift    = 8.0 if is_small_mark else 0.0
        decorated.append((box["xmin"] + mark_shift, box["cy"], idx))

    decorated.sort()
    return [segments[idx] for _, _, idx in decorated]


def merge_segments_human_writing_order(
    segments: list,
    max_endpoint_distance: float = 28.0,
    max_angle: float = 125.0,
    bridge_points: int = 5,
    min_branch_length: float = 0.0,
    keep_closed_cycles: bool = True,
    open_closed_cycles: bool = True,
    cycle_connection_distance: float = 24.0,
    protect_short_isolated: bool = True,
    expected_letter_gap: float = None,
    distance_weight: float = 1.10,
    angle_weight: float = 0.70,
    letter_weight: float = 0.55,
    forward_weight: float = 0.30,
    length_bonus: float = 0.08,
    relaxed_close_distance: float = 7.0,
    relaxed_max_angle: float = 155.0,
    verbose: bool = True,
) -> list:
    """
    Fusiona segmentos intentando aproximar la forma en que una persona escribiría.

    Diferencias clave respecto a merge_segments_by_best_continuation:
    - Empieza por trazos de izquierda a derecha, no por el segmento más largo.
    - Permite completar vueltas o lazos antes de continuar.
    - Evita puentes largos entre letras alejadas.
    - Conserva puntos, tildes y trazos pequeños como trayectorias separadas.
    - Ordena las trayectorias finales en orden de escritura.

    Parámetros:
        max_endpoint_distance  : distancia máxima para unir extremos (px).
        max_angle              : ángulo máximo de cambio de dirección (°).
        bridge_points          : puntos interpolados al unir extremos separados.
        expected_letter_gap    : None = estimación automática por tamaño de letra.
        verbose                : imprime estadísticas de la fusión.
    """
    cleaned          = []
    closed_segments  = []

    for seg in segments:
        seg = _remove_consecutive_duplicates(np.asarray(seg, dtype=float))
        if len(seg) < 2 or segment_length(seg) < 1e-6:
            continue

        if keep_closed_cycles and is_closed_segment(seg):
            closed_segments.append(seg)
            continue

        if min_branch_length > 0 and segment_length(seg) < min_branch_length:
            if protect_short_isolated:
                cleaned.append(seg)
        else:
            cleaned.append(seg)

    isolated_cycles = closed_segments

    if keep_closed_cycles and open_closed_cycles:
        isolated_cycles, opened_cycles = _open_cycles_near_open_segments(
            closed_segments, cleaned, connection_distance=cycle_connection_distance
        )
        cleaned = cleaned + opened_cycles

    if not cleaned:
        result = _sort_human_writing_order(isolated_cycles)
        if verbose:
            print("Modo escritura humana: no hay segmentos abiertos; se conservan ciclos aislados.")
        return result

    # Estimar un tamaño típico de letra a partir de las cajas de los segmentos.
    widths  = np.array([max(_segment_bbox(s)["w"], 1.0) for s in cleaned], dtype=float)
    heights = np.array([max(_segment_bbox(s)["h"], 1.0) for s in cleaned], dtype=float)
    if expected_letter_gap is None:
        expected_letter_gap = float(max(16.0, 0.65 * np.median(widths) + 0.25 * np.median(heights)))

    # Orden inicial: izquierda a derecha.
    order = sorted(
        range(len(cleaned)),
        key=lambda i: (_segment_bbox(cleaned[i])["xmin"], _segment_bbox(cleaned[i])["cy"])
    )
    used   = [False] * len(cleaned)
    merged = []

    for i in order:
        if used[i]:
            continue

        current  = _orient_segment_for_human_start(cleaned[i])
        used[i]  = True

        while True:
            choice = _choose_next_human_like(
                current               = current,
                segments              = cleaned,
                used                  = used,
                max_endpoint_distance = max_endpoint_distance,
                max_angle             = max_angle,
                expected_letter_gap   = expected_letter_gap,
                distance_weight       = distance_weight,
                angle_weight          = angle_weight,
                letter_weight         = letter_weight,
                forward_weight        = forward_weight,
                length_bonus          = length_bonus,
                relaxed_close_distance = relaxed_close_distance,
                relaxed_max_angle     = relaxed_max_angle,
            )

            if choice is None:
                break

            j, mode, _ = choice
            current    = _join_segments_with_bridge(current, cleaned[j], mode, bridge_points)
            used[j]    = True

        merged.append(current)

    result = isolated_cycles + merged
    result = _sort_human_writing_order(result)

    if verbose:
        print("Modo escritura humana activado")
        print(f"Separación típica estimada entre trazos/letras: {expected_letter_gap:.2f} px")

    return result


# ===========================================================================
# 9. Segmentos → trayectorias spline
# ===========================================================================

def segments_to_spline_trajectories(
    segments: list,
    smoothing: float = 2.0,
    num_points: int = 150,
    min_points: int = 4,
) -> list:
    """
    Convierte segmentos ordenados del esqueleto en trayectorias spline.

    Si hay pocos puntos, usa interpolación lineal para no perder trazos.
    """
    trajectories = []

    for segment in segments:
        segment = _remove_consecutive_duplicates(np.asarray(segment, dtype=float))
        if len(segment) < 2:
            continue

        x, y = segment[:, 0], segment[:, 1]

        if len(segment) < min_points:
            t     = np.linspace(0, 1, len(segment))
            t_new = np.linspace(0, 1, num_points)
            trajectories.append(np.column_stack((np.interp(t_new, t, x), np.interp(t_new, t, y))))
            continue

        try:
            closed = is_closed_segment(segment, distance_threshold=5)
            k      = min(3, len(segment) - 1)
            tck, _ = splprep([x, y], s=smoothing, per=closed, k=k)
            x_new, y_new = splev(np.linspace(0, 1, num_points), tck)
            trajectories.append(np.column_stack((x_new, y_new)))

        except Exception as e:
            t     = np.linspace(0, 1, len(segment))
            t_new = np.linspace(0, 1, num_points)
            trajectories.append(np.column_stack((np.interp(t_new, t, x), np.interp(t_new, t, y))))
            print(f"Spline falló; se usó interpolación lineal: {e}")

    return trajectories
