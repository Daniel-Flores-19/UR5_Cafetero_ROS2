#!/usr/bin/env python3
"""
letter_trajectory_functions.py — pipeline V7: texto -> trayectorias sobre A4.

Flujo:
    texto -> imagen A4 -> binarización/marcas -> esqueleto -> caminos
    -> orden/fusión con retroceso local/global -> splines -> mm A4

Uso:
    splines, splines_mm, data, graph_info = run_full_pipeline(texto, fuente, **overrides)

Todos los parámetros del pipeline viven en DEFAULTS: cualquier clave de ese
diccionario se puede pasar como keyword a run_full_pipeline().
"""

from __future__ import annotations

import os
from collections import deque
from itertools import count
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

NEIGHBOR_OFFSETS_8 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]

# Metadata por defecto de todo camino. Evita repetir claves en cada creación.
PATH_META = {
    "closed": False, "is_mark": False, "is_dot": False, "is_diaeresis": False,
    "is_accent": False, "mark_kind": None, "source": "skeleton_path",
    "merged_count": 1, "retrace_count": 0, "global_retrace_count": 0,
}

DEFAULTS = {
    # Lienzo A4, ROI de escritura y fuente
    "output_dir": "letras_png_A4",
    "px_per_mm": 6.0,
    "roi_x_mm": 0.0, "roi_y_mm": 249.0, "roi_w_mm": 210.0, "roi_h_mm": 46.0,
    "roi_margin_mm": 2.0,
    "max_font_size": 0,              # 0 = calcular automáticamente
    # Binarización
    "threshold_value": 200,
    "close_kernel_size": 5, "close_iterations": 1,
    "erosion_kernel_size": 3, "erosion_iterations": 0,
    # Marcas superiores: puntos de i/j, diéresis, tildes
    "detect_marks": True,
    "mark_min_area": 20, "mark_max_area": 1200,
    "mark_min_size": 4, "mark_max_size": 75,
    "mark_upper_region_quantile": 0.72,
    "dot_max_aspect_ratio": 1.65, "dot_max_eccentricity": 0.78,
    "diaeresis_pair_min_dx": 8, "diaeresis_pair_max_dx": 58, "diaeresis_pair_max_dy": 18,
    "dot_spline_points": 70, "dot_radius_scale": 0.85,
    "accent_min_path_pixels": 4,
    # Esqueleto y orden de escritura
    "min_path_pixels": 12,
    "order_strategy": "left_to_right",   # o "nearest"
    "postpone_marks_to_end": True,
    # Fusión de caminos
    "merge_close_paths": True,
    "merge_max_gap_pixels": 30, "merge_max_vertical_gap_pixels": 50,
    "merge_connector_points": 8,
    "avoid_merging_marks": True,
    "merge_with_retrace": True,
    "retrace_attach_max_distance": 24,
    "retrace_max_backtrack_pixels": 260,
    "retrace_connector_points": 4,
    "global_retrace_enabled": True,
    "global_retrace_attach_max_distance": 24,
    "global_retrace_max_route_pixels": 420,
    "global_retrace_connector_points": 4,
    "global_retrace_allow_marks": False,
    # Splines
    "spline_smoothing": 2.0, "spline_points": 180, "spline_degree": 3,
}


# =============================================================================
# Configuración
# =============================================================================

def make_config(**overrides) -> SimpleNamespace:
    """Combina DEFAULTS con los overrides y agrega los valores derivados."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"Parámetros desconocidos: {sorted(unknown)}")
    c = SimpleNamespace(**{**DEFAULTS, **overrides})
    c.roi_mm = (float(c.roi_x_mm), float(c.roi_y_mm), float(c.roi_w_mm), float(c.roi_h_mm))
    c.margin_px = int(round(c.roi_margin_mm * c.px_per_mm))
    c.image_size = (int(round(A4_WIDTH_MM * c.px_per_mm)), int(round(A4_HEIGHT_MM * c.px_per_mm)))
    return c


def _make_path(pixels, **meta) -> dict:
    """Crea un camino {pixels [y,x] + metadata}."""
    return {"pixels": np.asarray(pixels), **PATH_META, **meta}


def _xy(path) -> np.ndarray:
    """Puntos del camino como [x, y] (los pixels se guardan como [y, x])."""
    return np.asarray(path["pixels"], dtype=float)[:, ::-1]


# =============================================================================
# Imagen del texto sobre la hoja A4
# =============================================================================

FONT_FALLBACKS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]


def get_valid_font_path(font_path: str) -> str:
    """Retorna la fuente pedida, o la primera TrueType de respaldo disponible."""
    for candidate in [font_path, *FONT_FALLBACKS]:
        if candidate and os.path.exists(candidate):
            if candidate != font_path:
                print(f"Advertencia: fuente no encontrada. Se usa: {candidate}")
            return candidate
    try:
        from matplotlib import font_manager  # importación opcional
        return font_manager.findfont("DejaVu Sans")
    except Exception as exc:
        raise FileNotFoundError(
            "No se encontró la fuente indicada ni una fuente TrueType de respaldo."
        ) from exc


def get_next_filename(output_dir: str, text: str, font_path: str) -> str:
    """Nombre incremental para no sobrescribir imágenes anteriores."""
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{str(text).replace(' ', '_')}_{os.path.basename(font_path).split('.')[0]}"
    for i in count(1):
        path = os.path.join(output_dir, f"{stem}_{i}.png")
        if not os.path.exists(path):
            return path


def render_text_image(text: str, font_path: str, c: SimpleNamespace) -> str:
    """
    Dibuja el texto en negro sobre una hoja A4 blanca, centrado en el ROI.

    El ROI no se dibuja: cualquier trazo extra contaminaría la binarización.
    La escala px/mm es la misma en X e Y, así el esqueleto no se deforma.
    """
    x_mm, y_mm, w_mm, h_mm = c.roi_mm
    if min(x_mm, y_mm) < 0 or min(w_mm, h_mm) <= 0 \
            or x_mm + w_mm > A4_WIDTH_MM or y_mm + h_mm > A4_HEIGHT_MM:
        raise ValueError(f"ROI inválido o fuera de la hoja A4: {c.roi_mm}")

    font_path = get_valid_font_path(font_path)
    file_path = get_next_filename(c.output_dir, text, font_path)
    img = Image.new("RGB", c.image_size, "white")
    draw = ImageDraw.Draw(img)

    to_px = lambda v: int(round(float(v) * float(c.px_per_mm)))
    x0 = to_px(x_mm) + c.margin_px
    y0 = to_px(y_mm) + c.margin_px
    usable_w = max(1, to_px(w_mm) - 2 * c.margin_px)
    usable_h = max(1, to_px(h_mm) - 2 * c.margin_px)

    size = int(c.max_font_size) if c.max_font_size else max(12, int(round(usable_h * 1.8)))
    while size > 10:
        bbox = draw.textbbox((0, 0), text, font=ImageFont.truetype(font_path, size))
        if bbox[2] - bbox[0] <= usable_w and bbox[3] - bbox[1] <= usable_h:
            break
        size -= 1

    font = ImageFont.truetype(font_path, size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        (x0 + (usable_w - text_w) / 2 - bbox[0], y0 + (usable_h - text_h) / 2 - bbox[1]),
        text, fill="black", font=font,
    )
    img.save(file_path)
    print(f"Imagen guardada en: {file_path}")
    print(f"Lienzo {c.image_size[0]}x{c.image_size[1]} px a {c.px_per_mm:.3f} px/mm | "
          f"ROI {c.roi_mm} mm | margen {c.margin_px} px | fuente {size} px | "
          f"texto {text_w}x{text_h} px")
    return file_path


# =============================================================================
# Binarización y marcas superiores (puntos de i/j, diéresis, tildes)
# =============================================================================

def _pca(points: np.ndarray):
    """Media, eigenvalores (mayor primero) y eigenvectores de una nube Nx2."""
    mean = points.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(points - mean, rowvar=False))
    order = np.argsort(eigvals)[::-1]
    return mean, eigvals[order], eigvecs[:, order]


def _eccentricity(mask: np.ndarray) -> float:
    """Excentricidad del componente: 0 es un disco, cerca de 1 es alargado."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return 0.0
    _, eigvals, _ = _pca(np.column_stack((xs, ys)).astype(float))
    major, minor = max(eigvals[0], 1e-9), max(eigvals[1], 1e-9)
    return float(np.sqrt(max(0.0, 1.0 - minor / major)))


def _is_diaeresis_pair(a: dict, b: dict, c: SimpleNamespace) -> bool:
    """Dos puntos compactos, próximos y casi alineados forman una diéresis."""
    dx = abs(b["centroid_xy"][0] - a["centroid_xy"][0])
    dy = abs(b["centroid_xy"][1] - a["centroid_xy"][1])
    if not c.diaeresis_pair_min_dx <= dx <= c.diaeresis_pair_max_dx or dy > c.diaeresis_pair_max_dy:
        return False
    (_, _, wa, ha), (_, _, wb, hb) = a["bbox"], b["bbox"]
    return (max(wa, wb) / max(min(wa, wb), 1) <= 2.0
            and max(ha, hb) / max(min(ha, hb), 1) <= 2.0)


def detect_upper_mark_components(binary_clean: np.ndarray, c: SimpleNamespace):
    """
    Detecta marcas superiores separadas del cuerpo de la palabra:
    dot_i_j (punto compacto), diaeresis_dot (par de puntos) y accent_or_tilde.

    No se asume que todo componente pequeño superior sea un punto: primero se
    analiza su forma (relación de aspecto + excentricidad por PCA).
    """
    binary = (binary_clean > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    foreground_y = np.nonzero(binary)[0]
    if len(foreground_y) == 0:
        return [], labels

    y_limit = np.quantile(foreground_y, c.mark_upper_region_quantile)
    marks = []
    for label in range(1, num_labels):
        x, y, w, h, area = (int(v) for v in stats[label, :5])
        cx, cy = centroids[label]
        if not c.mark_min_area <= area <= c.mark_max_area:
            continue
        if min(w, h) < c.mark_min_size or max(w, h) > c.mark_max_size or cy > y_limit:
            continue
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        compact = (aspect_ratio <= c.dot_max_aspect_ratio
                   and _eccentricity(labels[y:y + h, x:x + w] == label) <= c.dot_max_eccentricity)
        marks.append({
            "label": label, "bbox": (x, y, w, h), "area": area,
            "centroid_xy": np.array([cx, cy], dtype=float),
            "aspect_ratio": float(aspect_ratio),
            "is_compact_dot_candidate": bool(compact),
            "mark_kind": "dot_i_j" if compact else "accent_or_tilde",
        })

    compact_indices = [i for i, m in enumerate(marks) if m["is_compact_dot_candidate"]]
    paired: set[int] = set()
    for i in compact_indices:
        if i in paired:
            continue
        partners = [j for j in compact_indices
                    if j > i and j not in paired and _is_diaeresis_pair(marks[i], marks[j], c)]
        if partners:
            j = min(partners, key=lambda k: abs(marks[k]["centroid_xy"][0] - marks[i]["centroid_xy"][0]))
            marks[i]["mark_kind"] = marks[j]["mark_kind"] = "diaeresis_dot"
            paired.update((i, j))

    marks.sort(key=lambda m: (m["centroid_xy"][0], m["centroid_xy"][1]))
    return marks, labels


def remove_mark_components_from_skeleton(skeleton_bool, mark_components, padding=2):
    """Borra del esqueleto las zonas de las marcas para no duplicarlas."""
    out = skeleton_bool.copy()
    height, width = out.shape
    for comp in mark_components:
        x, y, w, h = comp["bbox"]
        out[max(y - padding, 0):min(y + h + padding, height),
            max(x - padding, 0):min(x + w + padding, width)] = False
    return out


def _ellipse_pixels(comp: dict, c: SimpleNamespace) -> np.ndarray:
    """Trayectoria cerrada tipo elipse para un punto de i/j o de diéresis."""
    _, _, w, h = comp["bbox"]
    cx, cy = comp["centroid_xy"]
    rx = max(w * 0.5 * c.dot_radius_scale, 2.0)
    ry = max(h * 0.5 * c.dot_radius_scale, 2.0)
    theta = np.linspace(0, 2 * np.pi, c.dot_spline_points)
    return np.column_stack((cy + ry * np.sin(theta), cx + rx * np.cos(theta)))


def _accent_pixels(comp: dict, labels: np.ndarray, c: SimpleNamespace) -> list:
    """
    Forma real de una tilde/acento a partir del esqueleto local del componente.
    Si ese esqueleto queda demasiado corto, se aproxima con su eje principal.
    """
    x, y, w, h = comp["bbox"]
    local = labels[y:y + h, x:x + w] == comp["label"]
    local_paths, _ = skeleton_to_paths(skeletonize(local), min_path_pixels=c.accent_min_path_pixels)
    if local_paths:
        return [p["pixels"].astype(float) + (y, x) for p in local_paths]

    ys, xs = np.nonzero(local)
    if len(xs) < 2:
        return [np.array([comp["centroid_xy"][::-1]], dtype=float)]
    points = np.column_stack((xs + x, ys + y)).astype(float)
    mean, _, eigvecs = _pca(points)
    axis = eigvecs[:, 0]
    projections = (points - mean) @ axis
    line_xy = np.linspace(mean + projections.min() * axis, mean + projections.max() * axis, 20)
    return [line_xy[:, ::-1]]


def mark_components_to_paths(mark_components, labels, c: SimpleNamespace) -> list:
    """Convierte las marcas detectadas en caminos dibujables."""
    paths = []
    for mark_id, comp in enumerate(mark_components, start=1):
        kind = comp["mark_kind"]
        if kind in ("dot_i_j", "diaeresis_dot"):
            paths.append(_make_path(
                _ellipse_pixels(comp, c), closed=True, is_mark=True,
                is_dot=kind == "dot_i_j", is_diaeresis=kind == "diaeresis_dot",
                mark_kind=kind, source=kind, mark_id=mark_id, component=comp,
            ))
        else:
            for part, pixels in enumerate(_accent_pixels(comp, labels, c), start=1):
                paths.append(_make_path(
                    pixels, is_mark=True, is_accent=True, mark_kind="accent_or_tilde",
                    source="accent_or_tilde", mark_id=mark_id, accent_part=part, component=comp,
                ))
    return paths


def preprocess_text_image(image_path: str, c: SimpleNamespace) -> dict:
    """Binariza la imagen, detecta las marcas superiores y obtiene el esqueleto."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, c.threshold_value, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        np.ones((c.close_kernel_size, c.close_kernel_size), np.uint8),
        iterations=c.close_iterations,
    )
    if c.erosion_iterations > 0:
        binary = cv2.erode(
            binary, np.ones((c.erosion_kernel_size, c.erosion_kernel_size), np.uint8),
            iterations=c.erosion_iterations,
        )

    if c.detect_marks:
        mark_components, labels = detect_upper_mark_components(binary, c)
    else:
        mark_components = []
        labels = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)[1]

    skeleton = skeletonize(binary > 0)
    return {
        "binary_clean": binary,
        "skeleton_bool": skeleton,
        "skeleton_no_marks_bool": remove_mark_components_from_skeleton(skeleton, mark_components),
        "mark_components": mark_components,
        "connected_labels": labels,
    }


# =============================================================================
# Esqueleto -> caminos ordenables
# =============================================================================

def get_pixel_neighbors(pixel, skeleton_pixels) -> list:
    """Vecinos 8-conectados de un píxel del esqueleto."""
    y, x = pixel
    return [(y + dy, x + dx) for dy, dx in NEIGHBOR_OFFSETS_8 if (y + dy, x + dx) in skeleton_pixels]


def edge_key(pixel_a, pixel_b):
    """Identificador único de la arista entre dos píxeles."""
    return (pixel_a, pixel_b) if pixel_a <= pixel_b else (pixel_b, pixel_a)


def _trace(start, first_step, skeleton_pixels, graph_nodes, visited_edges, closed):
    """
    Recorre el esqueleto desde `start`.

    closed=False: avanza hasta el siguiente nodo del grafo (extremo o bifurcación).
    closed=True: sigue el ciclo hasta volver al inicio (letras como o, a, O).
    """
    path = [start]
    previous, current = start, first_step
    visited_edges.add(edge_key(start, first_step))
    for _ in range(len(skeleton_pixels) + 10):
        path.append(current)
        if closed:
            if current == start and len(path) > 2:
                break
        elif current in graph_nodes and current != start:
            break
        neighbors = [n for n in get_pixel_neighbors(current, skeleton_pixels) if n != previous]
        unvisited = [n for n in neighbors if edge_key(current, n) not in visited_edges]
        if closed and start in neighbors and len(path) > 3:
            nxt = start
        elif unvisited:
            nxt = unvisited[0]
        else:
            break
        visited_edges.add(edge_key(current, nxt))
        previous, current = current, nxt
    return np.array(path, dtype=np.int32)


def skeleton_to_paths(skeleton_bool, min_path_pixels: int = 12):
    """
    Convierte el esqueleto en caminos ordenados {pixels [y,x], closed}.

    Primero los tramos entre nodos del grafo, luego los ciclos que quedaron sin
    recorrer. Retorna también los extremos y bifurcaciones del esqueleto.
    """
    skeleton_pixels = set(map(tuple, np.column_stack(np.nonzero(skeleton_bool))))
    degrees = {p: len(get_pixel_neighbors(p, skeleton_pixels)) for p in skeleton_pixels}
    graph_nodes = {p for p, d in degrees.items() if d != 2}
    graph_info = {
        "endpoints": np.array([p for p, d in degrees.items() if d == 1], np.int32).reshape(-1, 2),
        "junctions": np.array([p for p, d in degrees.items() if d > 2], np.int32).reshape(-1, 2),
    }

    visited_edges, paths = set(), []
    for closed, starts in ((False, sorted(graph_nodes)), (True, sorted(skeleton_pixels))):
        for start in starts:
            for neighbor in get_pixel_neighbors(start, skeleton_pixels):
                if edge_key(start, neighbor) in visited_edges:
                    continue
                pixels = _trace(start, neighbor, skeleton_pixels, graph_nodes, visited_edges, closed)
                if len(pixels) >= min_path_pixels:
                    paths.append(_make_path(pixels, closed=closed))
    return paths, graph_info


# =============================================================================
# Orden de escritura
# =============================================================================

def reverse_path(path: dict) -> dict:
    """Invierte el sentido de un camino preservando su metadata."""
    return {**path, "pixels": np.asarray(path["pixels"])[::-1].copy()}


def orient_path_near_reference(path: dict, reference_xy=None) -> dict:
    """
    Orienta el camino para empezar por el extremo más cercano a la referencia.
    Sin referencia se empieza por el extremo más a la izquierda. Los caminos
    cerrados se rotan para arrancar en su punto más cercano.
    """
    xy = _xy(path)
    if reference_xy is None:
        return reverse_path(path) if xy[-1][0] < xy[0][0] else path
    if path["closed"]:
        idx = int(np.argmin(np.linalg.norm(xy - reference_xy, axis=1)))
        return {**path, "pixels": np.roll(np.asarray(path["pixels"]), -idx, axis=0)}
    start_dist = np.linalg.norm(xy[0] - reference_xy)
    end_dist = np.linalg.norm(xy[-1] - reference_xy)
    return reverse_path(path) if end_dist < start_dist else path


def _mark_priority(path: dict) -> int:
    """El cuerpo de la letra se escribe antes que sus marcas superiores."""
    if not path["is_mark"]:
        return 0
    return {"accent_or_tilde": 1, "diaeresis_dot": 2, "dot_i_j": 3}.get(path["mark_kind"], 4)


def order_paths_left_to_right(paths: list) -> list:
    """Ordena de izquierda a derecha, dejando cada marca tras su trazo base."""
    def sort_key(path):
        xy = _xy(path)
        x_key = xy[:, 0].mean() if path["is_mark"] or path["closed"] else xy[:, 0].min()
        return (x_key, _mark_priority(path), xy[:, 1].mean())

    ordered, previous_end = [], None
    for path in sorted(paths, key=sort_key):
        oriented = orient_path_near_reference(path, previous_end)
        ordered.append(oriented)
        previous_end = _xy(oriented)[-1]
    return ordered


def order_paths_nearest_neighbor(paths: list) -> list:
    """Ordena por vecino más cercano, empezando por el trazo más a la izquierda."""
    remaining = list(paths)
    if not remaining:
        return []
    first = remaining.pop(int(np.argmin([_xy(p)[:, 0].min() for p in remaining])))
    ordered = [orient_path_near_reference(first, None)]
    current_end = _xy(ordered[-1])[-1]
    while remaining:
        def cost(path):
            xy = _xy(path)
            if path["closed"]:
                return np.min(np.linalg.norm(xy - current_end, axis=1))
            return min(np.linalg.norm(xy[0] - current_end), np.linalg.norm(xy[-1] - current_end))

        nearest = remaining.pop(int(np.argmin([cost(p) for p in remaining])))
        ordered.append(orient_path_near_reference(nearest, current_end))
        current_end = _xy(ordered[-1])[-1]
    return ordered


def order_paths_for_writing(paths: list, strategy: str = "left_to_right") -> list:
    """Selecciona la estrategia de ordenamiento de caminos."""
    if strategy == "left_to_right":
        return order_paths_left_to_right(paths)
    if strategy == "nearest":
        return order_paths_nearest_neighbor(paths)
    raise ValueError("strategy debe ser 'left_to_right' o 'nearest'")


# =============================================================================
# Fusión de caminos: conector directo, retroceso local y retroceso global
# =============================================================================

def _connector(point_a, point_b, num_points: int) -> np.ndarray:
    """Puntos intermedios rectos entre dos posiciones."""
    if num_points <= 0:
        return np.empty((0, 2), dtype=float)
    return np.linspace(np.asarray(point_a, float), np.asarray(point_b, float), num_points + 2)[1:-1]


def _merged_path(path_a, path_b, pixels, source, retrace=0, global_retrace=0) -> dict:
    """Camino resultante de unir A con B, con la metadata de fusión al día."""
    merged = {**path_a, "pixels": np.asarray(pixels, dtype=float)}
    merged.update(
        closed=False, is_mark=False, is_dot=False, is_diaeresis=False, is_accent=False,
        mark_kind=None, source=source,
        merged_count=path_a["merged_count"] + path_b["merged_count"],
        retrace_count=path_a["retrace_count"] + path_b["retrace_count"] + retrace,
        global_retrace_count=path_a["global_retrace_count"] + path_b["global_retrace_count"] + global_retrace,
    )
    return merged


def can_merge_paths(path_a: dict, path_b: dict, c: SimpleNamespace) -> bool:
    """
    Dos caminos se pueden unir con una recta corta si están próximos.
    Las marcas y los ciclos cerrados nunca se fusionan: deformaría su forma.
    """
    if path_a["is_mark"] or path_b["is_mark"] or path_a["closed"] or path_b["closed"]:
        return False
    end_a, start_b = _xy(path_a)[-1], _xy(path_b)[0]
    return (np.linalg.norm(start_b - end_a) <= c.merge_max_gap_pixels
            and abs(start_b[1] - end_a[1]) <= c.merge_max_vertical_gap_pixels)


def _cumulative_length(pixels) -> np.ndarray:
    """Longitud acumulada del camino, en píxeles."""
    steps = np.linalg.norm(np.diff(np.asarray(pixels, dtype=float), axis=0), axis=1)
    return np.insert(np.cumsum(steps), 0, 0.0)


def find_retrace_candidate(current: dict, next_path: dict, c: SimpleNamespace):
    """
    Busca un punto anterior del camino actual al que volver para enlazar el
    siguiente trazo: simula que el lápiz retrocede en vez de saltar en recto.
    """
    if current["is_mark"] or next_path["is_mark"] or current["closed"] or next_path["closed"]:
        return None
    xy = _xy(current)
    cumulative = _cumulative_length(current["pixels"])
    best = None
    for candidate in (next_path, reverse_path(next_path)):
        distances = np.linalg.norm(xy - _xy(candidate)[0], axis=1)
        attach_idx = int(np.argmin(distances))
        backtrack = float(cumulative[-1] - cumulative[attach_idx])
        if attach_idx >= len(xy) - 3 or distances[attach_idx] > c.retrace_attach_max_distance:
            continue
        if backtrack > c.retrace_max_backtrack_pixels:
            continue
        cost = float(distances[attach_idx]) + 0.15 * backtrack
        if best is None or cost < best["cost"]:
            best = {"path": candidate, "attach_idx": attach_idx, "cost": cost}
    return best


def add_retrace_connector(current: dict, next_path: dict, attach_idx: int, c: SimpleNamespace) -> dict:
    """Termina el trazo actual, retrocede hasta attach_idx y sigue con el siguiente."""
    pixels, next_pixels = current["pixels"], next_path["pixels"]
    retrace = pixels[attach_idx:-1][::-1]
    connector = _connector(pixels[attach_idx], next_pixels[0], c.retrace_connector_points)
    return _merged_path(
        current, next_path, np.vstack([pixels, retrace, connector, next_pixels]),
        "merged_retrace_path", retrace=1,
    )


def update_visited_ink(visited_ink: set, path: dict) -> set:
    """Registra como tinta dibujada todos los píxeles enteros que cubre el camino."""
    pixels = np.asarray(path["pixels"], dtype=float)
    if len(pixels) == 1:
        visited_ink.add(tuple(int(v) for v in np.rint(pixels[0])))
    for start, end in zip(pixels[:-1], pixels[1:]):
        steps = max(int(np.ceil(float(np.linalg.norm(end - start)))), 1)
        for point in np.rint(np.linspace(start, end, steps + 1)).astype(int):
            visited_ink.add((int(point[0]), int(point[1])))
    return visited_ink


def _shortest_path_on_visited_ink(start_pixel, target_pixels, visited_ink, max_route_pixels):
    """BFS en vecindad 8 sobre la tinta ya dibujada. None si no hay ruta."""
    if start_pixel not in visited_ink or not target_pixels:
        return None
    if start_pixel in target_pixels:
        return [start_pixel]
    parent = {start_pixel: None}
    queue = deque([(start_pixel, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth >= max_route_pixels:
            continue
        y, x = current
        for dy, dx in NEIGHBOR_OFFSETS_8:
            neighbor = (y + dy, x + dx)
            if neighbor not in visited_ink or neighbor in parent:
                continue
            parent[neighbor] = current
            if neighbor in target_pixels:
                route = [neighbor]
                while parent[route[-1]] is not None:
                    route.append(parent[route[-1]])
                return route[::-1]
            queue.append((neighbor, depth + 1))
    return None


def find_global_retrace_route(current: dict, next_path: dict, visited_ink: set, c: SimpleNamespace):
    """
    Ruta desde el final del camino actual hasta el inicio del siguiente caminando
    sobre cualquier trazo ya escrito (no solo sobre el camino actual).
    """
    if not visited_ink:
        return None
    ink = np.array(list(visited_ink), dtype=float)
    max_attach = c.global_retrace_attach_max_distance

    distances = np.linalg.norm(ink - np.asarray(current["pixels"][-1], float), axis=1)
    nearest = int(np.argmin(distances))
    if distances[nearest] > max(3.0, max_attach):
        return None
    start_pixel = tuple(ink[nearest].astype(int))

    next_start = np.asarray(next_path["pixels"][0], dtype=float)
    targets = set(map(tuple, ink[np.linalg.norm(ink - next_start, axis=1) <= max_attach].astype(int)))
    route = _shortest_path_on_visited_ink(
        start_pixel, targets, visited_ink, c.global_retrace_max_route_pixels)
    if route is None:
        return None
    target_distance = float(np.linalg.norm(np.asarray(route[-1], dtype=float) - next_start))
    return {"route_pixels": np.array(route, dtype=float),
            "cost": float(len(route) + 2.0 * target_distance)}


def find_best_global_retrace_candidate(current: dict, next_path: dict, visited_ink: set, c: SimpleNamespace):
    """Prueba el siguiente camino en ambos sentidos y escoge la ruta más barata."""
    if not c.global_retrace_allow_marks and (current["is_mark"] or next_path["is_mark"]):
        return None
    best = None
    candidates = [next_path] if next_path["closed"] else [next_path, reverse_path(next_path)]
    for candidate in candidates:
        route_info = find_global_retrace_route(current, candidate, visited_ink, c)
        if route_info is not None and (best is None or route_info["cost"] < best["cost"]):
            best = {"path": candidate, "route_pixels": route_info["route_pixels"],
                    "cost": route_info["cost"]}
    return best


def add_global_retrace_connector(current: dict, next_path: dict, route_pixels, c: SimpleNamespace) -> dict:
    """Une los dos caminos insertando una ruta que ya estaba dibujada."""
    pixels = np.asarray(current["pixels"], dtype=float)
    next_pixels = np.asarray(next_path["pixels"], dtype=float)
    route = np.asarray(route_pixels, dtype=float)

    if len(route) == 0:
        route_to_add, route_end = np.empty((0, 2), dtype=float), pixels[-1]
    else:
        route_to_add = route[1:] if np.linalg.norm(route[0] - pixels[-1]) < 1e-9 else route
        route_end = route[-1]

    gap = float(np.linalg.norm(route_end - next_pixels[0]))
    connector = (np.empty((0, 2), dtype=float) if gap < 1e-9
                 else _connector(route_end, next_pixels[0], c.global_retrace_connector_points))
    return _merged_path(
        current, next_path, np.vstack([pixels, route_to_add, connector, next_pixels]),
        "merged_global_retrace_path", retrace=1, global_retrace=1,
    )


def merge_close_ordered_paths(ordered_paths: list, c: SimpleNamespace) -> list:
    """
    Fusiona caminos consecutivos en tres niveles:
    1. conector recto si están muy próximos,
    2. retroceso local sobre el camino actual,
    3. retroceso global sobre cualquier tinta ya dibujada.

    Si ninguno aplica, los caminos quedan separados: eso es un pen-up real.
    """
    if not ordered_paths:
        return []
    merged, visited_ink = [], set()
    current = ordered_paths[0]
    update_visited_ink(visited_ink, current)

    for next_path in ordered_paths[1:]:
        next_path = orient_path_near_reference(next_path, _xy(current)[-1])
        skip_mark = c.avoid_merging_marks and (current["is_mark"] or next_path["is_mark"])

        if not skip_mark and can_merge_paths(current, next_path, c):
            pixels = np.vstack([current["pixels"],
                                _connector(current["pixels"][-1], next_path["pixels"][0],
                                           c.merge_connector_points),
                                next_path["pixels"]])
            current = _merged_path(current, next_path, pixels, "merged_path")
        elif c.merge_with_retrace and (local := find_retrace_candidate(current, next_path, c)):
            current = add_retrace_connector(current, local["path"], local["attach_idx"], c)
        elif c.global_retrace_enabled and (
                glob := find_best_global_retrace_candidate(current, next_path, visited_ink, c)):
            current = add_global_retrace_connector(current, glob["path"], glob["route_pixels"], c)
        else:
            merged.append(current)
            current = next_path
        update_visited_ink(visited_ink, current)

    merged.append(current)
    return merged


# =============================================================================
# Splines
# =============================================================================

def _remove_consecutive_duplicates(points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) <= 1:
        return points_xy
    keep = np.ones(len(points_xy), dtype=bool)
    keep[1:] = np.any(np.diff(points_xy, axis=0) != 0, axis=1)
    return points_xy[keep]


def linear_resample(points_xy: np.ndarray, num_points: int = 120) -> np.ndarray:
    """Re-muestreo lineal de respaldo, cuando no alcanza para un spline."""
    points_xy = _remove_consecutive_duplicates(points_xy)
    if len(points_xy) <= 1:
        return points_xy
    distance = np.insert(np.cumsum(np.linalg.norm(np.diff(points_xy, axis=0), axis=1)), 0, 0)
    if distance[-1] == 0:
        return points_xy
    new_distance = np.linspace(0, distance[-1], num_points)
    return np.column_stack([np.interp(new_distance, distance, points_xy[:, i]) for i in (0, 1)])


def fit_spline_to_path(points_xy, smoothing=2.0, num_points=120, degree=3, closed=False) -> np.ndarray:
    """Ajusta un spline 2D a un camino [x, y]; cae a interpolación lineal si falla."""
    points_xy = _remove_consecutive_duplicates(np.asarray(points_xy, dtype=float))
    if len(points_xy) < 2:
        return points_xy
    if closed and np.linalg.norm(points_xy[0] - points_xy[-1]) > 1e-6:
        points_xy = np.vstack([points_xy, points_xy[0]])
    k = min(degree, len(points_xy) - 1)
    if k < 2:
        return linear_resample(points_xy, num_points=num_points)
    try:
        tck, _ = splprep([points_xy[:, 0], points_xy[:, 1]], s=smoothing, k=k, per=closed)
        x_new, y_new = splev(np.linspace(0, 1, num_points), tck)
        return np.column_stack((x_new, y_new))
    except Exception as error:
        print(f"No se pudo ajustar spline. Se usa interpolación lineal. Error: {error}")
        return linear_resample(points_xy, num_points=num_points)


def paths_to_splines(ordered_paths: list, c: SimpleNamespace) -> list:
    """Convierte los caminos finales en splines, preservando su metadata."""
    splines = []
    for idx, path in enumerate(ordered_paths, start=1):
        control_points = _xy(path)
        spline = {key: path[key] for key in
                  ("closed", "is_mark", "is_dot", "is_diaeresis", "is_accent", "mark_kind",
                   "source", "merged_count", "retrace_count", "global_retrace_count")}
        spline.update(
            id=idx,
            control_points=control_points,
            points=fit_spline_to_path(control_points, smoothing=c.spline_smoothing,
                                      num_points=c.spline_points, degree=c.spline_degree,
                                      closed=path["closed"]),
        )
        splines.append(spline)
    return splines


# =============================================================================
# Pipeline completo V7
# =============================================================================

def run_full_pipeline(text: str, font_path: str, **overrides):
    """
    Ejecuta el pipeline completo. Acepta como keywords cualquier clave de DEFAULTS.

    Retorna:
        splines: lista de dicts con puntos en píxeles, en mm y metadata.
        spline_arrays_mm: lista de arrays Nx2 en milímetros sobre la hoja A4.
        data: imágenes intermedias, marcas y caminos de cada etapa.
        graph_info: extremos y bifurcaciones del esqueleto sin marcas.
    """
    c = make_config(**overrides)

    image_path = render_text_image(text, font_path, c)
    data = preprocess_text_image(image_path, c)

    mark_paths = mark_components_to_paths(data["mark_components"], data["connected_labels"], c)
    skeleton_paths, graph_info = skeleton_to_paths(data["skeleton_no_marks_bool"], c.min_path_pixels)

    if c.postpone_marks_to_end:
        ordered_paths = (order_paths_for_writing(skeleton_paths, c.order_strategy)
                         + order_paths_for_writing(mark_paths, "left_to_right"))
    else:
        ordered_paths = order_paths_for_writing(skeleton_paths + mark_paths, c.order_strategy)

    final_paths = merge_close_ordered_paths(ordered_paths, c) if c.merge_close_paths else ordered_paths

    splines = paths_to_splines(final_paths, c)
    # px -> mm: misma escala en X e Y, sin normalizar ni estirar.
    spline_arrays_mm = [spline["points"] / float(c.px_per_mm) for spline in splines]
    for spline, points_mm in zip(splines, spline_arrays_mm):
        spline["points_mm"] = points_mm

    data.update(
        image_path=image_path,
        raw_skeleton_paths=skeleton_paths,
        mark_paths=mark_paths,
        raw_paths=skeleton_paths + mark_paths,
        ordered_paths=ordered_paths,
        final_paths=final_paths,
        spline_arrays_px=[spline["points"] for spline in splines],
        spline_arrays_mm=spline_arrays_mm,
        roi_mm=c.roi_mm,
        px_per_mm=float(c.px_per_mm),
    )
    return splines, spline_arrays_mm, data, graph_info


__all__ = ["run_full_pipeline", "make_config", "DEFAULTS", "A4_WIDTH_MM", "A4_HEIGHT_MM"]
