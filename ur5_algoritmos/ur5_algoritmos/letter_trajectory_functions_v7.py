#!/usr/bin/env python3
"""
letter_trajectory_functions_v7.py

Funciones del pipeline V7 para convertir texto manuscrito en trayectorias.

Flujo:
    texto -> imagen A4 -> binarización/marcas -> esqueleto -> caminos
    -> orden/fusión con retroceso local/global -> splines -> mm A4

Esta versión viene del notebook letras_a_trayectorias_jupyter_V7 y queda
lista para usarse desde array_new_v7.py o desde un nodo ROS2.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev


# =============================================================================
# Constantes principales V7
# =============================================================================

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
PX_PER_MM = 6.0
CANVAS_WIDTH_PX = int(round(A4_WIDTH_MM * PX_PER_MM))
CANVAS_HEIGHT_PX = int(round(A4_HEIGHT_MM * PX_PER_MM))
IMAGE_SIZE = (CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX)

ROI_X_MM = 0.0
ROI_Y_MM = 249.0
ROI_W_MM = 210.0
ROI_H_MM = 46.0
ROI_MARGIN_MM = 2.0
ROI_MM = (ROI_X_MM, ROI_Y_MM, ROI_W_MM, ROI_H_MM)
MARGIN = int(round(ROI_MARGIN_MM * PX_PER_MM))

DEFAULT_OUTPUT_DIR = "letras_png_A4"

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
DOT_SPLINE_POINTS = 70
DOT_RADIUS_SCALE = 0.85
DIAERESIS_PAIR_MIN_DX = 8
DIAERESIS_PAIR_MAX_DX = 58
DIAERESIS_PAIR_MAX_DY = 18
ACCENT_MIN_PATH_PIXELS = 4

MIN_PATH_PIXELS = 12
ORDER_STRATEGY = "left_to_right"

MERGE_CLOSE_PATHS = True
MERGE_MAX_GAP_PIXELS = 30
MERGE_MAX_VERTICAL_GAP_PIXELS = 50
MERGE_CONNECTOR_POINTS = 8
MERGE_WITH_RETRACE = True
RETRACE_ATTACH_MAX_DISTANCE = 24
RETRACE_MAX_BACKTRACK_PIXELS = 260
RETRACE_CONNECTOR_POINTS = 4

GLOBAL_RETRACE_ENABLED = True
GLOBAL_RETRACE_ATTACH_MAX_DISTANCE = 24
GLOBAL_RETRACE_MAX_ROUTE_PIXELS = 420
GLOBAL_RETRACE_CONNECTOR_POINTS = 4
GLOBAL_RETRACE_ALLOW_MARKS = False

POSTPONE_MARKS_TO_END = True
AVOID_MERGING_MARKS = True

SPLINE_SMOOTHING = 2.0
SPLINE_POINTS = 180
SPLINE_DEGREE = 3

NEIGHBOR_OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


# =============================================================================
# Fuente tipográfica
# =============================================================================

def get_valid_font_path(font_path: str) -> str:
    """
    Retorna la fuente indicada si existe.

    Si la ruta no existe, busca fuentes TrueType comunes. Se evita depender de
    matplotlib para que el módulo pueda correr en entornos ROS2 más livianos.
    """
    if font_path and os.path.exists(font_path):
        return font_path

    fallback_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in fallback_candidates:
        if os.path.exists(candidate):
            print("Advertencia: no se encontró la fuente indicada.")
            print(f"Usando fuente de respaldo: {candidate}")
            return candidate

    try:
        from matplotlib import font_manager  # importación opcional
        fallback = font_manager.findfont("DejaVu Sans")
        print("Advertencia: no se encontró la fuente indicada.")
        print(f"Usando fuente de respaldo: {fallback}")
        return fallback
    except Exception as exc:
        raise FileNotFoundError(
            "No se encontró la fuente indicada ni una fuente TrueType de respaldo. "
            "Instala DejaVuSans.ttf o pasa una ruta válida en font_path."
        ) from exc

def get_next_filename(output_dir, text, font_path):
    """Genera un nombre incremental para no sobrescribir imágenes anteriores."""
    os.makedirs(output_dir, exist_ok=True)
    safe_text = str(text).replace(' ', '_')
    font_name = os.path.basename(font_path).split('.')[0]
    i = 1
    while True:
        file_name = f'{safe_text}_{font_name}_{i}.png'
        file_path = os.path.join(output_dir, file_name)
        if not os.path.exists(file_path):
            return file_path
        i += 1

def image_size_to_wh(image_size):
    """
    Permite usar image_size como entero cuadrado o como tupla (width_px, height_px).
    """
    if isinstance(image_size, (tuple, list, np.ndarray)):
        return (int(image_size[0]), int(image_size[1]))
    return (int(image_size), int(image_size))

def mm_to_px(value_mm, px_per_mm=PX_PER_MM):
    """Convierte milímetros a píxeles usando la misma escala en X e Y."""
    return int(round(float(value_mm) * float(px_per_mm)))

def roi_mm_to_px(roi_mm, px_per_mm=PX_PER_MM):
    """
    Convierte un ROI físico (x_mm, y_mm, w_mm, h_mm) a píxeles.
    """
    x_mm, y_mm, w_mm, h_mm = roi_mm
    return (mm_to_px(x_mm, px_per_mm), mm_to_px(y_mm, px_per_mm), mm_to_px(w_mm, px_per_mm), mm_to_px(h_mm, px_per_mm))

def validate_a4_roi(roi_mm, a4_width_mm=A4_WIDTH_MM, a4_height_mm=A4_HEIGHT_MM):
    """Verifica que el recuadro esté dentro de la hoja A4."""
    x_mm, y_mm, w_mm, h_mm = roi_mm
    if x_mm < 0 or y_mm < 0:
        raise ValueError('El ROI no puede empezar fuera de la hoja A4.')
    if w_mm <= 0 or h_mm <= 0:
        raise ValueError('El ROI debe tener ancho y alto positivos.')
    if x_mm + w_mm > a4_width_mm:
        raise ValueError('El ROI se sale del ancho de la hoja A4.')
    if y_mm + h_mm > a4_height_mm:
        raise ValueError('El ROI se sale del alto de la hoja A4.')

def points_px_to_a4_mm(points_xy, px_per_mm=PX_PER_MM):
    """
    Convierte puntos [x_px, y_px] a [x_mm, y_mm] sobre la hoja A4.

    Importante:
    - No normaliza a [0, 1].
    - No estira X ni Y.
    - Solo divide por la misma escala px/mm.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    return points_xy / float(px_per_mm)

def splines_px_to_a4_mm(splines, px_per_mm=PX_PER_MM):
    """Convierte todos los splines de píxeles a milímetros."""
    return [points_px_to_a4_mm(spline['points'], px_per_mm) for spline in splines]

def render_text_image(text, font_path, output_dir='letras_png_A4', image_size=IMAGE_SIZE, max_font_size=None, margin=0, roi_mm=ROI_MM, px_per_mm=PX_PER_MM):
    """
    Genera una imagen RGB con fondo blanco y texto negro sobre una hoja A4.

    La palabra se ubica dentro del ROI físico indicado en milímetros.
    La imagen final se procesa directamente con esa geometría; no se reescala
    de forma no uniforme, por lo que el esqueleto no se deforma.

    El recuadro rojo NO se dibuja en la imagen, porque si se dibujara afectaría
    la binarización y el skeletonization.
    """
    validate_a4_roi(roi_mm)
    font_path = get_valid_font_path(font_path)
    file_path = get_next_filename(output_dir, text, font_path)
    image_w_px, image_h_px = image_size_to_wh(image_size)
    img = Image.new('RGB', (image_w_px, image_h_px), 'white')
    draw = ImageDraw.Draw(img)
    roi_x_px, roi_y_px, roi_w_px, roi_h_px = roi_mm_to_px(roi_mm, px_per_mm)
    margin_px = int(round(margin))
    usable_x_px = roi_x_px + margin_px
    usable_y_px = roi_y_px + margin_px
    usable_w_px = max(1, roi_w_px - 2 * margin_px)
    usable_h_px = max(1, roi_h_px - 2 * margin_px)
    if max_font_size is None:
        font_size = max(12, int(round(usable_h_px * 1.8)))
    else:
        font_size = int(max_font_size)
    while font_size > 10:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= usable_w_px and text_h <= usable_h_px:
            break
        font_size -= 1
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = usable_x_px + (usable_w_px - text_w) / 2 - bbox[0]
    y = usable_y_px + (usable_h_px - text_h) / 2 - bbox[1]
    draw.text((x, y), text, fill='black', font=font)
    img.save(file_path)
    print(f'Imagen guardada en: {file_path}')
    print(f'Hoja A4: {A4_WIDTH_MM:.1f} mm x {A4_HEIGHT_MM:.1f} mm')
    print(f'Lienzo: {image_w_px} px x {image_h_px} px')
    print(f'Escala: {px_per_mm:.3f} px/mm')
    print(f'Recuadro ROI [mm]: x={roi_mm[0]}, y={roi_mm[1]}, w={roi_mm[2]}, h={roi_mm[3]}')
    print(f'Recuadro ROI [px]: x={roi_x_px}, y={roi_y_px}, w={roi_w_px}, h={roi_h_px}')
    print(f'Margen interno: {margin_px} px = {margin_px / px_per_mm:.2f} mm')
    print(f'Tamaño de fuente usado: {font_size}')
    print(f'BBox del texto [px]: w={text_w}, h={text_h}')
    return file_path

def _component_pca_features(component_mask):
    """
    Calcula rasgos geométricos simples de un componente conectado.
    Retorna excentricidad, ángulo principal y eigenvalues de PCA.
    """
    ys, xs = np.where(component_mask > 0)
    if len(xs) < 3:
        return {'eccentricity': 0.0, 'angle_deg': 0.0, 'lambda_major': 0.0, 'lambda_minor': 0.0}
    points = np.column_stack((xs, ys)).astype(float)
    points_centered = points - np.mean(points, axis=0)
    cov = np.cov(points_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    lambda_major = float(max(eigvals[0], 1e-09))
    lambda_minor = float(max(eigvals[1], 1e-09))
    eccentricity = float(np.sqrt(max(0.0, 1.0 - lambda_minor / lambda_major)))
    angle_rad = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    angle_deg = float(np.degrees(angle_rad))
    return {'eccentricity': eccentricity, 'angle_deg': angle_deg, 'lambda_major': lambda_major, 'lambda_minor': lambda_minor}

def _is_diaeresis_pair(comp_a, comp_b, min_dx=8, max_dx=58, max_dy=18):
    """
    Decide si dos componentes compactos pueden formar una diéresis.
    La diéresis se ve como dos puntos superiores próximos y casi alineados.
    """
    cxa, cya = comp_a['centroid_xy']
    cxb, cyb = comp_b['centroid_xy']
    dx = abs(cxb - cxa)
    dy = abs(cyb - cya)
    if dx < min_dx or dx > max_dx:
        return False
    if dy > max_dy:
        return False
    _, _, wa, ha = comp_a['bbox']
    _, _, wb, hb = comp_b['bbox']
    size_ratio_w = max(wa, wb) / max(min(wa, wb), 1)
    size_ratio_h = max(ha, hb) / max(min(ha, hb), 1)
    if size_ratio_w > 2.0 or size_ratio_h > 2.0:
        return False
    return True

def detect_upper_mark_components(binary_clean, min_area=20, max_area=1200, min_size=4, max_size=75, upper_region_quantile=0.72, dot_max_aspect_ratio=1.65, dot_max_eccentricity=0.78, diaeresis_pair_min_dx=8, diaeresis_pair_max_dx=58, diaeresis_pair_max_dy=18):
    """
    Detecta marcas superiores separadas de la palabra:
    - dot_i_j: punto compacto de i/j.
    - diaeresis_dot: puntos dobles de diéresis.
    - accent_or_tilde: tilde, acento agudo u otra marca alargada.

    La diferencia principal respecto a la V4 inicial es que ya no se asume que
    todo componente pequeño superior es un punto. Primero se analiza su forma.
    """
    binary_uint8 = (binary_clean > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_uint8, connectivity=8)
    foreground_y = np.where(binary_uint8 > 0)[0]
    if len(foreground_y) == 0:
        return ([], labels)
    y_limit = np.quantile(foreground_y, upper_region_quantile)
    candidates = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]
        if area < min_area or area > max_area:
            continue
        if w < min_size or h < min_size:
            continue
        if w > max_size or h > max_size:
            continue
        if cy > y_limit:
            continue
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        component_mask = labels[y:y + h, x:x + w] == label
        pca = _component_pca_features(component_mask)
        is_compact_dot = aspect_ratio <= dot_max_aspect_ratio and pca['eccentricity'] <= dot_max_eccentricity
        candidates.append({'label': label, 'bbox': (x, y, w, h), 'area': area, 'centroid_xy': np.array([cx, cy], dtype=float), 'aspect_ratio': float(aspect_ratio), 'eccentricity': pca['eccentricity'], 'angle_deg': pca['angle_deg'], 'is_compact_dot_candidate': bool(is_compact_dot), 'mark_kind': 'dot_i_j' if is_compact_dot else 'accent_or_tilde'})
    compact_indices = [i for i, comp in enumerate(candidates) if comp['is_compact_dot_candidate']]
    used = set()
    for i in compact_indices:
        if i in used:
            continue
        best_j = None
        best_dx = np.inf
        for j in compact_indices:
            if j <= i or j in used:
                continue
            if _is_diaeresis_pair(candidates[i], candidates[j], min_dx=diaeresis_pair_min_dx, max_dx=diaeresis_pair_max_dx, max_dy=diaeresis_pair_max_dy):
                dx = abs(candidates[j]['centroid_xy'][0] - candidates[i]['centroid_xy'][0])
                if dx < best_dx:
                    best_dx = dx
                    best_j = j
        if best_j is not None:
            pair_id = len(used) + 1
            candidates[i]['mark_kind'] = 'diaeresis_dot'
            candidates[best_j]['mark_kind'] = 'diaeresis_dot'
            candidates[i]['diaeresis_pair_id'] = pair_id
            candidates[best_j]['diaeresis_pair_id'] = pair_id
            used.add(i)
            used.add(best_j)
    candidates = sorted(candidates, key=lambda comp: (comp['centroid_xy'][0], comp['centroid_xy'][1]))
    return (candidates, labels)

def remove_mark_components_from_skeleton(skeleton_bool, mark_components, padding=2):
    """
    Elimina del esqueleto las zonas donde se detectaron marcas superiores.
    Así evitamos duplicar puntos, diéresis y acentos.
    """
    skeleton_no_marks = skeleton_bool.copy()
    height, width = skeleton_no_marks.shape
    for comp in mark_components:
        x, y, w, h = comp['bbox']
        x0 = max(x - padding, 0)
        y0 = max(y - padding, 0)
        x1 = min(x + w + padding, width)
        y1 = min(y + h + padding, height)
        skeleton_no_marks[y0:y1, x0:x1] = False
    return skeleton_no_marks

def _ellipse_path_from_component(comp, num_points=70, radius_scale=0.85):
    """Crea una trayectoria cerrada tipo elipse para puntos de i/j o diéresis."""
    x, y, w, h = comp['bbox']
    cx, cy = comp['centroid_xy']
    rx = max(w * 0.5 * radius_scale, 2.0)
    ry = max(h * 0.5 * radius_scale, 2.0)
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
    xs = cx + rx * np.cos(theta)
    ys = cy + ry * np.sin(theta)
    return np.column_stack((ys, xs)).astype(float)

def _pca_line_path_from_component(comp, labels, num_points=20):
    """
    Respaldo para marcas muy pequeñas: aproxima el componente con una línea
    siguiendo su eje principal.
    """
    label = comp['label']
    x, y, w, h = comp['bbox']
    local = labels[y:y + h, x:x + w] == label
    ys, xs = np.where(local > 0)
    if len(xs) == 0:
        cx, cy = comp['centroid_xy']
        return np.array([[cy, cx]], dtype=float)
    points = np.column_stack((xs + x, ys + y)).astype(float)
    if len(points) < 2:
        cx, cy = comp['centroid_xy']
        return np.array([[cy, cx]], dtype=float)
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    projections = centered @ direction
    p0 = mean + projections.min() * direction
    p1 = mean + projections.max() * direction
    line_xy = np.linspace(p0, p1, num_points)
    return np.column_stack((line_xy[:, 1], line_xy[:, 0])).astype(float)

def _accent_path_from_component(comp, labels, min_path_pixels=4):
    """
    Conserva la forma real de una tilde/acento a partir del esqueleto local
    del componente. Si el esqueleto queda demasiado corto, usa PCA como respaldo.
    """
    label = comp['label']
    x, y, w, h = comp['bbox']
    local_mask = labels[y:y + h, x:x + w] == label
    local_skeleton = skeletonize(local_mask > 0)
    local_paths, _ = skeleton_to_paths(local_skeleton, min_path_pixels=min_path_pixels)
    if len(local_paths) == 0:
        pixels = _pca_line_path_from_component(comp, labels)
        return [pixels]
    paths = []
    for path in local_paths:
        local_pixels = path['pixels'].astype(float)
        local_pixels[:, 0] += y
        local_pixels[:, 1] += x
        paths.append(local_pixels)
    return paths

def mark_components_to_paths(mark_components, labels, dot_num_points=70, dot_radius_scale=0.85, accent_min_path_pixels=4):
    """
    Convierte marcas superiores detectadas en caminos:
    - puntos i/j: elipse cerrada.
    - diéresis: dos elipses cerradas, etiquetadas como diéresis.
    - tilde/acento: forma del esqueleto real del componente.
    """
    mark_paths = []
    for idx, comp in enumerate(mark_components):
        mark_kind = comp.get('mark_kind', 'unknown')
        if mark_kind in ['dot_i_j', 'diaeresis_dot']:
            pixels = _ellipse_path_from_component(comp, num_points=dot_num_points, radius_scale=dot_radius_scale)
            mark_paths.append({'pixels': pixels, 'closed': True, 'is_mark': True, 'is_dot': mark_kind == 'dot_i_j', 'is_diaeresis': mark_kind == 'diaeresis_dot', 'is_accent': False, 'mark_kind': mark_kind, 'source': mark_kind, 'mark_id': idx + 1, 'component': comp})
        else:
            accent_paths = _accent_path_from_component(comp, labels, min_path_pixels=accent_min_path_pixels)
            for k, pixels in enumerate(accent_paths):
                mark_paths.append({'pixels': pixels, 'closed': False, 'is_mark': True, 'is_dot': False, 'is_diaeresis': False, 'is_accent': True, 'mark_kind': 'accent_or_tilde', 'source': 'accent_or_tilde', 'mark_id': idx + 1, 'accent_part': k + 1, 'component': comp})
    return mark_paths

def preprocess_text_image(image_path, threshold_value=200, close_kernel_size=5, close_iterations=1, erosion_kernel_size=3, erosion_iterations=0, detect_marks=True, mark_min_area=20, mark_max_area=1200, mark_min_size=4, mark_max_size=75, mark_upper_region_quantile=0.72, dot_max_aspect_ratio=1.65, dot_max_eccentricity=0.78, diaeresis_pair_min_dx=8, diaeresis_pair_max_dx=58, diaeresis_pair_max_dy=18):
    """
    Lee la imagen, la binariza, detecta marcas superiores y obtiene el esqueleto.

    Retorna un diccionario con:
    - image_rgb
    - gray
    - binary_clean
    - skeleton_bool
    - skeleton_no_marks_bool
    - skeleton_uint8
    - skeleton_overlay
    - mark_components
    - connected_labels
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f'No se pudo cargar la imagen: {image_path}')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)
    if erosion_iterations > 0:
        erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        binary_clean = cv2.erode(binary_clean, erosion_kernel, iterations=erosion_iterations)
    mark_components = []
    connected_labels = None
    if detect_marks:
        mark_components, connected_labels = detect_upper_mark_components(binary_clean, min_area=mark_min_area, max_area=mark_max_area, min_size=mark_min_size, max_size=mark_max_size, upper_region_quantile=mark_upper_region_quantile, dot_max_aspect_ratio=dot_max_aspect_ratio, dot_max_eccentricity=dot_max_eccentricity, diaeresis_pair_min_dx=diaeresis_pair_min_dx, diaeresis_pair_max_dx=diaeresis_pair_max_dx, diaeresis_pair_max_dy=diaeresis_pair_max_dy)
    else:
        connected_labels = cv2.connectedComponentsWithStats((binary_clean > 0).astype(np.uint8), connectivity=8)[1]
    skeleton_bool = skeletonize(binary_clean > 0)
    skeleton_no_marks_bool = remove_mark_components_from_skeleton(skeleton_bool, mark_components)
    skeleton_uint8 = (skeleton_bool * 255).astype(np.uint8)
    skeleton_overlay = img_rgb.copy()
    y_coords, x_coords = np.where(skeleton_bool)
    skeleton_overlay[y_coords, x_coords] = [255, 0, 0]
    return {'image_rgb': img_rgb, 'gray': gray, 'binary_clean': binary_clean, 'skeleton_bool': skeleton_bool, 'skeleton_no_marks_bool': skeleton_no_marks_bool, 'skeleton_uint8': skeleton_uint8, 'skeleton_overlay': skeleton_overlay, 'mark_components': mark_components, 'connected_labels': connected_labels}

def get_skeleton_pixels(skeleton_bool):
    """Retorna los píxeles activos del esqueleto como set de tuplas (y, x)."""
    coords = np.column_stack(np.where(skeleton_bool > 0))
    return set(map(tuple, coords))

def get_pixel_neighbors(pixel, skeleton_pixels):
    """Obtiene vecinos 8-conectados de un píxel del esqueleto."""
    y, x = pixel
    neighbors = []
    for dy, dx in NEIGHBOR_OFFSETS_8:
        candidate = (y + dy, x + dx)
        if candidate in skeleton_pixels:
            neighbors.append(candidate)
    return neighbors

def classify_skeleton_pixels(skeleton_bool):
    """
    Clasifica los píxeles del esqueleto en:
    - endpoints
    - junctions
    - normal_points

    También retorna degree_map, útil para trazar caminos.
    """
    skeleton_pixels = get_skeleton_pixels(skeleton_bool)
    endpoints = []
    junctions = []
    normal_points = []
    isolated_points = []
    degree_map = {}
    for pixel in skeleton_pixels:
        degree = len(get_pixel_neighbors(pixel, skeleton_pixels))
        degree_map[pixel] = degree
        if degree == 0:
            isolated_points.append(pixel)
        elif degree == 1:
            endpoints.append(pixel)
        elif degree == 2:
            normal_points.append(pixel)
        else:
            junctions.append(pixel)
    return {'skeleton_pixels': skeleton_pixels, 'degree_map': degree_map, 'endpoints': np.array(endpoints, dtype=np.int32), 'junctions': np.array(junctions, dtype=np.int32), 'normal_points': np.array(normal_points, dtype=np.int32), 'isolated_points': np.array(isolated_points, dtype=np.int32)}

def edge_key(pixel_a, pixel_b):
    """Identificador único de una arista entre dos píxeles."""
    return tuple(sorted((pixel_a, pixel_b)))

def trace_path_from_node(start_pixel, next_pixel, skeleton_pixels, degree_map, graph_nodes, visited_edges):
    """
    Traza un camino desde un nodo del grafo hasta otro nodo.
    Un nodo es un extremo, bifurcación o punto aislado.
    """
    path = [start_pixel]
    previous_pixel = start_pixel
    current_pixel = next_pixel
    visited_edges.add(edge_key(start_pixel, next_pixel))
    max_steps = len(skeleton_pixels) + 10
    steps = 0
    while steps < max_steps:
        path.append(current_pixel)
        if current_pixel in graph_nodes and current_pixel != start_pixel:
            break
        neighbors = get_pixel_neighbors(current_pixel, skeleton_pixels)
        candidates = [n for n in neighbors if n != previous_pixel and edge_key(current_pixel, n) not in visited_edges]
        if len(candidates) == 0:
            break
        next_candidate = candidates[0]
        previous_pixel, current_pixel = (current_pixel, next_candidate)
        visited_edges.add(edge_key(previous_pixel, current_pixel))
        steps += 1
    return np.array(path, dtype=np.int32)

def trace_closed_path(start_pixel, next_pixel, skeleton_pixels, visited_edges):
    """
    Traza un ciclo cerrado cuando el componente no tiene extremos.
    Esto es importante para letras como 'o', 'O', 'a', etc.
    """
    path = [start_pixel]
    previous_pixel = start_pixel
    current_pixel = next_pixel
    visited_edges.add(edge_key(start_pixel, next_pixel))
    max_steps = len(skeleton_pixels) + 10
    steps = 0
    while steps < max_steps:
        path.append(current_pixel)
        if current_pixel == start_pixel and len(path) > 2:
            break
        neighbors = get_pixel_neighbors(current_pixel, skeleton_pixels)
        candidates = [n for n in neighbors if n != previous_pixel]
        unvisited_candidates = [n for n in candidates if edge_key(current_pixel, n) not in visited_edges]
        if start_pixel in candidates and len(path) > 3:
            next_candidate = start_pixel
        elif len(unvisited_candidates) > 0:
            next_candidate = unvisited_candidates[0]
        else:
            break
        if edge_key(current_pixel, next_candidate) not in visited_edges:
            visited_edges.add(edge_key(current_pixel, next_candidate))
        previous_pixel, current_pixel = (current_pixel, next_candidate)
        steps += 1
    return np.array(path, dtype=np.int32)

def skeleton_to_paths(skeleton_bool, min_path_pixels=12):
    """
    Convierte el esqueleto en una lista de caminos ordenados.

    Retorna una lista de diccionarios:
    {
        "pixels": arreglo Nx2 con columnas [y, x],
        "closed": True/False
    }
    """
    graph_info = classify_skeleton_pixels(skeleton_bool)
    skeleton_pixels = graph_info['skeleton_pixels']
    degree_map = graph_info['degree_map']
    graph_nodes = {pixel for pixel, degree in degree_map.items() if degree != 2}
    visited_edges = set()
    paths = []
    for node in sorted(graph_nodes):
        neighbors = get_pixel_neighbors(node, skeleton_pixels)
        for neighbor in neighbors:
            if edge_key(node, neighbor) in visited_edges:
                continue
            path_pixels = trace_path_from_node(node, neighbor, skeleton_pixels, degree_map, graph_nodes, visited_edges)
            if len(path_pixels) >= min_path_pixels:
                paths.append({'pixels': path_pixels, 'closed': False})
    for pixel in sorted(skeleton_pixels):
        neighbors = get_pixel_neighbors(pixel, skeleton_pixels)
        for neighbor in neighbors:
            if edge_key(pixel, neighbor) in visited_edges:
                continue
            path_pixels = trace_closed_path(pixel, neighbor, skeleton_pixels, visited_edges)
            if len(path_pixels) >= min_path_pixels:
                paths.append({'pixels': path_pixels, 'closed': True})
    return (paths, graph_info)

def pixels_to_xy(path_pixels):
    """
    Convierte puntos de imagen [y, x] a puntos cartesianos de imagen [x, y].
    """
    return np.column_stack((path_pixels[:, 1], path_pixels[:, 0])).astype(float)

def path_start_xy(path):
    xy = pixels_to_xy(path['pixels'])
    return xy[0]

def path_end_xy(path):
    xy = pixels_to_xy(path['pixels'])
    return xy[-1]

def path_bbox_xy(path):
    xy = pixels_to_xy(path['pixels'])
    return {'min_x': float(np.min(xy[:, 0])), 'max_x': float(np.max(xy[:, 0])), 'min_y': float(np.min(xy[:, 1])), 'max_y': float(np.max(xy[:, 1])), 'center_x': float(np.mean(xy[:, 0])), 'center_y': float(np.mean(xy[:, 1]))}

def reverse_path(path):
    """Invierte el sentido de un camino preservando su metadata."""
    new_path = path.copy()
    new_path['pixels'] = path['pixels'][::-1].copy()
    return new_path

def rotate_closed_path_to_nearest_point(path, reference_xy):
    """
    Para caminos cerrados, cambia el punto de inicio al punto más cercano
    a una referencia dada.
    """
    pixels = path['pixels']
    xy = pixels_to_xy(pixels)
    distances = np.linalg.norm(xy - reference_xy, axis=1)
    idx = int(np.argmin(distances))
    rotated_pixels = np.roll(pixels, -idx, axis=0)
    new_path = path.copy()
    new_path['pixels'] = rotated_pixels
    return new_path

def orient_path_near_reference(path, reference_xy=None):
    """
    Orienta el camino:
    - Si no hay referencia, intenta iniciar por el extremo más a la izquierda.
    - Si hay referencia, inicia por el extremo más cercano.
    """
    xy = pixels_to_xy(path['pixels'])
    if path.get('closed', False) and reference_xy is not None:
        return rotate_closed_path_to_nearest_point(path, reference_xy)
    if reference_xy is None:
        start = xy[0]
        end = xy[-1]
        if end[0] < start[0]:
            return reverse_path(path)
        return path
    start_dist = np.linalg.norm(xy[0] - reference_xy)
    end_dist = np.linalg.norm(xy[-1] - reference_xy)
    if end_dist < start_dist:
        return reverse_path(path)
    return path

def order_paths_nearest_neighbor(paths):
    """
    Ordena caminos con una heurística de vecino más cercano.
    Esta era la lógica principal de la V3.
    """
    if len(paths) == 0:
        return []
    remaining = [p.copy() for p in paths]
    min_x_values = []
    for path in remaining:
        xy = pixels_to_xy(path['pixels'])
        min_x_values.append(np.min(xy[:, 0]))
    first_idx = int(np.argmin(min_x_values))
    current_path = remaining.pop(first_idx)
    current_path = orient_path_near_reference(current_path, reference_xy=None)
    ordered = [current_path]
    current_end = path_end_xy(current_path)
    while len(remaining) > 0:
        best_idx = None
        best_cost = np.inf
        for idx, path in enumerate(remaining):
            xy = pixels_to_xy(path['pixels'])
            if path.get('closed', False):
                cost = np.min(np.linalg.norm(xy - current_end, axis=1))
            else:
                cost_start = np.linalg.norm(xy[0] - current_end)
                cost_end = np.linalg.norm(xy[-1] - current_end)
                cost = min(cost_start, cost_end)
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        next_path = remaining.pop(best_idx)
        next_path = orient_path_near_reference(next_path, current_end)
        ordered.append(next_path)
        current_end = path_end_xy(next_path)
    return ordered

def _path_sort_priority(path):
    """
    Prioridad secundaria para ordenar marcas.
    La base de la letra se dibuja antes que sus marcas superiores.
    """
    if not path.get('is_mark', False):
        return 0
    mark_kind = path.get('mark_kind', '')
    if mark_kind == 'accent_or_tilde':
        return 1
    if mark_kind == 'diaeresis_dot':
        return 2
    if mark_kind == 'dot_i_j':
        return 3
    return 4

def order_paths_left_to_right(paths):
    """
    Ordena caminos priorizando escritura de izquierda a derecha.

    Además, las marcas superiores se colocan cerca de su zona x, pero después
    del trazo base de la letra cuando sus posiciones se superponen.
    """
    if len(paths) == 0:
        return []
    paths_with_bbox = []
    for path in paths:
        bbox = path_bbox_xy(path)
        paths_with_bbox.append((path, bbox))

    def sort_key(item):
        path, bbox = item
        is_mark = path.get('is_mark', False)
        is_closed = path.get('closed', False)
        x_key = bbox['center_x'] if is_mark or is_closed else bbox['min_x']
        y_key = bbox['center_y']
        return (x_key, _path_sort_priority(path), y_key)
    sorted_items = sorted(paths_with_bbox, key=sort_key)
    ordered = []
    previous_end = None
    for path, _ in sorted_items:
        oriented = orient_path_near_reference(path, previous_end)
        ordered.append(oriented)
        previous_end = path_end_xy(oriented)
    return ordered

def order_paths_for_writing(paths, strategy='left_to_right'):
    """Selecciona la estrategia de ordenamiento de caminos."""
    if strategy == 'nearest':
        return order_paths_nearest_neighbor(paths)
    if strategy == 'left_to_right':
        return order_paths_left_to_right(paths)
    raise ValueError("strategy debe ser 'left_to_right' o 'nearest'")

def add_linear_connector(path_a, path_b, num_points=8):
    """
    Une dos caminos agregando puntos lineales entre el final de A y el inicio de B.
    El conector solo se usa cuando la distancia entre trazos es pequeña.
    """
    a_pixels = path_a['pixels']
    b_pixels = path_b['pixels']
    end_a = a_pixels[-1]
    start_b = b_pixels[0]
    if num_points <= 0:
        connector = np.empty((0, 2), dtype=float)
    else:
        connector = np.linspace(end_a, start_b, num_points + 2)[1:-1]
    merged_pixels = np.vstack([a_pixels, connector, b_pixels])
    new_path = path_a.copy()
    new_path['pixels'] = merged_pixels
    new_path['closed'] = False
    new_path['is_mark'] = False
    new_path['is_dot'] = False
    new_path['is_diaeresis'] = False
    new_path['is_accent'] = False
    new_path['mark_kind'] = None
    new_path['source'] = 'merged_path'
    new_path['merged_count'] = path_a.get('merged_count', 1) + path_b.get('merged_count', 1)
    new_path['retrace_count'] = path_a.get('retrace_count', 0) + path_b.get('retrace_count', 0)
    return new_path

def should_skip_mark_merge(path_a, path_b, avoid_merging_marks=True):
    """
    Retorna True cuando alguno de los caminos es una marca superior
    y se quiere evitar conectarla al cuerpo de la letra.
    """
    if not avoid_merging_marks:
        return False
    return bool(path_a.get('is_mark', False) or path_b.get('is_mark', False))

def can_merge_paths(path_a, path_b, max_gap=30, max_vertical_gap=50):
    """
    Decide si dos caminos se pueden fusionar directamente.

    No se fusionan marcas superiores ni ciclos cerrados. Las marcas se mantienen
    como strokes independientes para no deformar puntos, diéresis ni acentos.
    """
    if path_a.get('is_mark', False) or path_b.get('is_mark', False):
        return False
    if path_a.get('closed', False) or path_b.get('closed', False):
        return False
    end_a = path_end_xy(path_a)
    start_b = path_start_xy(path_b)
    distance = np.linalg.norm(start_b - end_a)
    vertical_gap = abs(start_b[1] - end_a[1])
    if distance > max_gap:
        return False
    if vertical_gap > max_vertical_gap:
        return False
    return True

def _path_cumulative_length_pixels(path_pixels):
    """Longitud acumulada de un camino en unidades de píxel."""
    if len(path_pixels) <= 1:
        return np.array([0.0])
    diffs = np.diff(path_pixels.astype(float), axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.insert(np.cumsum(seg_lengths), 0, 0.0)

def find_retrace_candidate(current_path, next_path, attach_max_distance=16, max_backtrack_pixels=75):
    """
    Busca si next_path puede unirse a un punto anterior de current_path.

    La idea es simular que el lápiz regresa sobre una parte ya escrita para
    continuar con un trazo cercano, en vez de hacer un salto recto nuevo.
    """
    if current_path.get('is_mark', False) or next_path.get('is_mark', False):
        return None
    if current_path.get('closed', False) or next_path.get('closed', False):
        return None
    current_pixels = current_path['pixels']
    current_xy = pixels_to_xy(current_pixels)
    cumulative = _path_cumulative_length_pixels(current_pixels)
    total_length = cumulative[-1]
    best = None
    for candidate in [next_path, reverse_path(next_path)]:
        candidate_start = path_start_xy(candidate)
        distances = np.linalg.norm(current_xy - candidate_start, axis=1)
        attach_idx = int(np.argmin(distances))
        attach_distance = float(distances[attach_idx])
        backtrack_length = float(total_length - cumulative[attach_idx])
        if attach_idx >= len(current_pixels) - 3:
            continue
        if attach_distance > attach_max_distance:
            continue
        if backtrack_length > max_backtrack_pixels:
            continue
        cost = attach_distance + 0.15 * backtrack_length
        if best is None or cost < best['cost']:
            best = {'path': candidate, 'attach_idx': attach_idx, 'attach_distance': attach_distance, 'backtrack_length': backtrack_length, 'cost': cost}
    return best

def add_retrace_connector(current_path, next_path, attach_idx, connector_points=4):
    """
    Fusiona current_path con next_path regresando sobre el mismo camino actual.

    Secuencia:
    1. Termina current_path.
    2. Retrocede por los mismos puntos hasta attach_idx.
    3. Agrega un conector corto si todavía hay una pequeña separación.
    4. Continúa con next_path.
    """
    current_pixels = current_path['pixels']
    next_pixels = next_path['pixels']
    retrace_segment = current_pixels[attach_idx:-1][::-1]
    attach_point = current_pixels[attach_idx]
    next_start = next_pixels[0]
    if connector_points <= 0:
        connector = np.empty((0, 2), dtype=float)
    else:
        connector = np.linspace(attach_point, next_start, connector_points + 2)[1:-1]
    merged_pixels = np.vstack([current_pixels, retrace_segment, connector, next_pixels])
    new_path = current_path.copy()
    new_path['pixels'] = merged_pixels
    new_path['closed'] = False
    new_path['is_mark'] = False
    new_path['is_dot'] = False
    new_path['is_diaeresis'] = False
    new_path['is_accent'] = False
    new_path['mark_kind'] = None
    new_path['source'] = 'merged_retrace_path'
    new_path['merged_count'] = current_path.get('merged_count', 1) + next_path.get('merged_count', 1)
    new_path['retrace_count'] = current_path.get('retrace_count', 0) + next_path.get('retrace_count', 0) + 1
    return new_path

def _bresenham_like_segment_pixels(point_a, point_b):
    """
    Rasteriza un segmento entre dos puntos [y, x].
    No usa antialiasing: solo genera píxeles enteros conectados.
    """
    point_a = np.asarray(point_a, dtype=float)
    point_b = np.asarray(point_b, dtype=float)
    distance = float(np.linalg.norm(point_b - point_a))
    steps = max(int(np.ceil(distance)), 1)
    samples = np.linspace(point_a, point_b, steps + 1)
    samples = np.rint(samples).astype(int)
    pixels = []
    last = None
    for p in samples:
        item = (int(p[0]), int(p[1]))
        if item != last:
            pixels.append(item)
            last = item
    return pixels

def rasterize_path_pixels(path_pixels):
    """
    Convierte un camino, que puede tener puntos flotantes por conectores,
    en una lista de píxeles enteros conectados [y, x].
    """
    path_pixels = np.asarray(path_pixels, dtype=float)
    if len(path_pixels) == 0:
        return []
    if len(path_pixels) == 1:
        p = np.rint(path_pixels[0]).astype(int)
        return [(int(p[0]), int(p[1]))]
    rasterized = []
    for idx in range(len(path_pixels) - 1):
        segment = _bresenham_like_segment_pixels(path_pixels[idx], path_pixels[idx + 1])
        if idx > 0 and len(segment) > 0:
            segment = segment[1:]
        rasterized.extend(segment)
    return rasterized

def update_visited_ink(visited_ink, path):
    """
    Agrega al conjunto visited_ink todos los píxeles cubiertos por un path.
    visited_ink representa la tinta ya dibujada por el robot/lápiz.
    """
    for pixel in rasterize_path_pixels(path['pixels']):
        visited_ink.add(pixel)
    return visited_ink

def _nearest_visited_pixel(reference_yx, visited_ink, max_distance=np.inf):
    """
    Busca el píxel ya dibujado más cercano a una referencia [y, x].
    Se usa para enganchar el movimiento al trazo existente.
    """
    if len(visited_ink) == 0:
        return (None, np.inf)
    reference_yx = np.asarray(reference_yx, dtype=float)
    visited_array = np.array(list(visited_ink), dtype=float)
    distances = np.linalg.norm(visited_array - reference_yx, axis=1)
    idx = int(np.argmin(distances))
    best_distance = float(distances[idx])
    if best_distance > max_distance:
        return (None, best_distance)
    best_pixel = tuple(visited_array[idx].astype(int))
    return (best_pixel, best_distance)

def _visited_pixels_near_reference(reference_yx, visited_ink, max_distance):
    """
    Retorna todos los píxeles ya dibujados que están cerca de una referencia.
    Esos píxeles son candidatos de llegada para el retroceso global.
    """
    if len(visited_ink) == 0:
        return set()
    reference_yx = np.asarray(reference_yx, dtype=float)
    visited_array = np.array(list(visited_ink), dtype=float)
    distances = np.linalg.norm(visited_array - reference_yx, axis=1)
    mask = distances <= max_distance
    if not np.any(mask):
        return set()
    nearby = visited_array[mask].astype(int)
    return set(map(tuple, nearby))

def _shortest_path_on_visited_ink(start_pixel, target_pixels, visited_ink, max_route_pixels=180):
    """
    Calcula una ruta sobre tinta ya dibujada usando BFS en vecindad 8.

    start_pixel pertenece a visited_ink.
    target_pixels es un conjunto de píxeles de visited_ink cercanos al siguiente spline.
    Si no hay conexión física dentro de la tinta ya dibujada, retorna None.
    """
    from collections import deque
    if start_pixel not in visited_ink:
        return None
    if len(target_pixels) == 0:
        return None
    if start_pixel in target_pixels:
        return [start_pixel]
    queue = deque([start_pixel])
    parent = {start_pixel: None}
    depth = {start_pixel: 0}
    while queue:
        current = queue.popleft()
        current_depth = depth[current]
        if current_depth >= max_route_pixels:
            continue
        y, x = current
        for dy, dx in NEIGHBOR_OFFSETS_8:
            neighbor = (y + dy, x + dx)
            if neighbor not in visited_ink:
                continue
            if neighbor in parent:
                continue
            parent[neighbor] = current
            depth[neighbor] = current_depth + 1
            if neighbor in target_pixels:
                route = [neighbor]
                while parent[route[-1]] is not None:
                    route.append(parent[route[-1]])
                route.reverse()
                return route
            queue.append(neighbor)
    return None

def find_global_retrace_route(current_path, next_path, visited_ink, attach_max_distance=18, max_route_pixels=180, allow_marks=False):
    """
    Busca una ruta desde el final del camino actual hasta el inicio del siguiente,
    pero caminando por cualquier trazo ya dibujado.

    Diferencia clave respecto al retroceso local:
    - El retroceso local solo mira puntos anteriores dentro de current_path.
    - El retroceso global mira toda la tinta ya escrita: caminos anteriores,
      fusiones anteriores y el camino actual.
    """
    if not allow_marks:
        if current_path.get('is_mark', False) or next_path.get('is_mark', False):
            return None
    if len(visited_ink) == 0:
        return None
    current_end_yx = np.asarray(current_path['pixels'][-1], dtype=float)
    next_start_yx = np.asarray(next_path['pixels'][0], dtype=float)
    start_pixel, start_distance = _nearest_visited_pixel(current_end_yx, visited_ink, max_distance=max(3.0, attach_max_distance))
    if start_pixel is None:
        return None
    target_pixels = _visited_pixels_near_reference(next_start_yx, visited_ink, max_distance=attach_max_distance)
    route = _shortest_path_on_visited_ink(start_pixel, target_pixels, visited_ink, max_route_pixels=max_route_pixels)
    if route is None:
        return None
    target_pixel = route[-1]
    target_distance = float(np.linalg.norm(np.asarray(target_pixel, dtype=float) - next_start_yx))
    return {'route_pixels': np.array(route, dtype=float), 'start_pixel': start_pixel, 'target_pixel': target_pixel, 'start_distance': float(start_distance), 'target_distance': target_distance, 'route_length_pixels': len(route), 'cost': float(len(route) + 2.0 * target_distance)}

def find_best_global_retrace_candidate(current_path, next_path, visited_ink, attach_max_distance=18, max_route_pixels=180, allow_marks=False):
    """
    Prueba el siguiente camino en ambos sentidos y escoge la orientación que permite
    llegar mediante la ruta más corta sobre tinta ya dibujada.
    """
    best = None
    candidates = [next_path]
    if not next_path.get('closed', False):
        candidates.append(reverse_path(next_path))
    for candidate_path in candidates:
        route_info = find_global_retrace_route(current_path, candidate_path, visited_ink, attach_max_distance=attach_max_distance, max_route_pixels=max_route_pixels, allow_marks=allow_marks)
        if route_info is None:
            continue
        cost = route_info['cost']
        if best is None or cost < best['cost']:
            best = {'path': candidate_path, 'route_info': route_info, 'cost': cost}
    return best

def add_global_retrace_connector(current_path, next_path, route_pixels, connector_points=4):
    """
    Fusiona current_path con next_path insertando una ruta sobre tinta ya dibujada.

    Secuencia física:
    1. Termina el spline/camino actual.
    2. Recorre una ruta que ya estaba dibujada.
    3. Si queda una separación pequeña hasta el inicio del siguiente camino,
       agrega un conector corto.
    4. Continúa con el siguiente camino.
    """
    current_pixels = np.asarray(current_path['pixels'], dtype=float)
    next_pixels = np.asarray(next_path['pixels'], dtype=float)
    route_pixels = np.asarray(route_pixels, dtype=float)
    if len(route_pixels) > 0:
        if np.linalg.norm(route_pixels[0] - current_pixels[-1]) < 1e-09:
            route_to_add = route_pixels[1:]
        else:
            route_to_add = route_pixels
    else:
        route_to_add = np.empty((0, 2), dtype=float)
    if len(route_pixels) == 0:
        route_end = current_pixels[-1]
    else:
        route_end = route_pixels[-1]
    next_start = next_pixels[0]
    gap = float(np.linalg.norm(route_end - next_start))
    if connector_points <= 0 or gap < 1e-09:
        connector = np.empty((0, 2), dtype=float)
    else:
        connector = np.linspace(route_end, next_start, connector_points + 2)[1:-1]
    merged_pixels = np.vstack([current_pixels, route_to_add, connector, next_pixels])
    new_path = current_path.copy()
    new_path['pixels'] = merged_pixels
    new_path['closed'] = False
    new_path['is_mark'] = False
    new_path['is_dot'] = False
    new_path['is_diaeresis'] = False
    new_path['is_accent'] = False
    new_path['mark_kind'] = None
    new_path['source'] = 'merged_global_retrace_path'
    new_path['merged_count'] = current_path.get('merged_count', 1) + next_path.get('merged_count', 1)
    new_path['retrace_count'] = current_path.get('retrace_count', 0) + next_path.get('retrace_count', 0) + 1
    new_path['global_retrace_count'] = current_path.get('global_retrace_count', 0) + next_path.get('global_retrace_count', 0) + 1
    return new_path

def merge_close_ordered_paths(ordered_paths, max_gap=30, max_vertical_gap=50, connector_points=8, merge_with_retrace=True, retrace_attach_max_distance=24, retrace_max_backtrack_pixels=260, retrace_connector_points=4, global_retrace_enabled=True, global_retrace_attach_max_distance=24, global_retrace_max_route_pixels=420, global_retrace_connector_points=4, global_retrace_allow_marks=False, avoid_merging_marks=True):
    """
    Fusiona caminos consecutivos si están suficientemente cerca.

    Niveles de unión usados:
    1. Conector directo si dos caminos están muy próximos.
    2. Retroceso local: vuelve sobre una parte del camino actual.
    3. Retroceso global: vuelve sobre cualquier trazo ya dibujado, usando la tinta
       previa como un grafo y buscando una ruta conectada sobre ese grafo.

    Si no existe una ruta válida sobre tinta ya dibujada, el algoritmo conserva
    caminos separados. En ejecución robótica, eso equivale a un pen-up real.
    """
    if len(ordered_paths) == 0:
        return []
    merged_paths = []
    current = ordered_paths[0]
    visited_ink = set()
    update_visited_ink(visited_ink, current)
    for next_path in ordered_paths[1:]:
        next_path = orient_path_near_reference(next_path, path_end_xy(current))
        if not should_skip_mark_merge(current, next_path, avoid_merging_marks) and can_merge_paths(current, next_path, max_gap=max_gap, max_vertical_gap=max_vertical_gap):
            current = add_linear_connector(current, next_path, num_points=connector_points)
            update_visited_ink(visited_ink, current)
            continue
        if merge_with_retrace:
            candidate = find_retrace_candidate(current, next_path, attach_max_distance=retrace_attach_max_distance, max_backtrack_pixels=retrace_max_backtrack_pixels)
            if candidate is not None:
                current = add_retrace_connector(current, candidate['path'], attach_idx=candidate['attach_idx'], connector_points=retrace_connector_points)
                update_visited_ink(visited_ink, current)
                continue
        if global_retrace_enabled:
            global_candidate = find_best_global_retrace_candidate(current, next_path, visited_ink, attach_max_distance=global_retrace_attach_max_distance, max_route_pixels=global_retrace_max_route_pixels, allow_marks=global_retrace_allow_marks)
            if global_candidate is not None:
                current = add_global_retrace_connector(current, global_candidate['path'], global_candidate['route_info']['route_pixels'], connector_points=global_retrace_connector_points)
                update_visited_ink(visited_ink, current)
                continue
        merged_paths.append(current)
        current = next_path
        update_visited_ink(visited_ink, current)
    merged_paths.append(current)
    return merged_paths

def remove_consecutive_duplicates(points_xy):
    """Elimina puntos repetidos consecutivos."""
    if len(points_xy) <= 1:
        return points_xy
    keep = np.ones(len(points_xy), dtype=bool)
    keep[1:] = np.any(np.diff(points_xy, axis=0) != 0, axis=1)
    return points_xy[keep]

def linear_resample(points_xy, num_points=120):
    """
    Re-muestreo lineal de respaldo.
    Se usa cuando no hay suficientes puntos para spline.
    """
    points_xy = remove_consecutive_duplicates(points_xy)
    if len(points_xy) <= 1:
        return points_xy
    segment_lengths = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    distance = np.insert(np.cumsum(segment_lengths), 0, 0)
    if distance[-1] == 0:
        return points_xy
    new_distance = np.linspace(0, distance[-1], num_points)
    x_new = np.interp(new_distance, distance, points_xy[:, 0])
    y_new = np.interp(new_distance, distance, points_xy[:, 1])
    return np.column_stack((x_new, y_new))

def fit_spline_to_path(points_xy, smoothing=2.0, num_points=120, degree=3, closed=False):
    """
    Ajusta un spline 2D a un camino.

    Entrada:
    - points_xy: arreglo Nx2 con puntos [x, y]
    - smoothing: suavizado del spline
    - num_points: puntos finales a generar
    - degree: grado del spline
    - closed: True para caminos cerrados
    """
    points_xy = remove_consecutive_duplicates(points_xy)
    if len(points_xy) < 2:
        return points_xy
    if closed:
        if np.linalg.norm(points_xy[0] - points_xy[-1]) > 1e-06:
            points_xy = np.vstack([points_xy, points_xy[0]])
    k = min(degree, len(points_xy) - 1)
    if k < 2:
        return linear_resample(points_xy, num_points=num_points)
    try:
        tck, u = splprep([points_xy[:, 0], points_xy[:, 1]], s=smoothing, k=k, per=closed)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack((x_new, y_new))
    except Exception as error:
        print(f'No se pudo ajustar spline. Se usa interpolación lineal. Error: {error}')
        return linear_resample(points_xy, num_points=num_points)

def paths_to_splines(ordered_paths, smoothing=2.0, num_points=120, degree=3):
    """
    Convierte una lista de caminos ordenados en una lista de splines.
    Preserva metadata como marcas, fuente, cantidad de fusiones y retrocesos.
    """
    splines = []
    for idx, path in enumerate(ordered_paths):
        control_points_xy = pixels_to_xy(path['pixels'])
        spline_points_xy = fit_spline_to_path(control_points_xy, smoothing=smoothing, num_points=num_points, degree=degree, closed=path.get('closed', False))
        spline = {'id': idx + 1, 'control_points': control_points_xy, 'points': spline_points_xy, 'closed': path.get('closed', False), 'is_mark': path.get('is_mark', False), 'is_dot': path.get('is_dot', False), 'is_diaeresis': path.get('is_diaeresis', False), 'is_accent': path.get('is_accent', False), 'mark_kind': path.get('mark_kind', None), 'source': path.get('source', 'skeleton_path'), 'merged_count': path.get('merged_count', 1), 'retrace_count': path.get('retrace_count', 0)}
        splines.append(spline)
    return splines

# =============================================================================
# Pipeline completo V7
# =============================================================================

def run_full_pipeline(
    text: str,
    font_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    image_size: Sequence[int] = IMAGE_SIZE,
    max_font_size: int | None = None,
    margin: int = MARGIN,
    roi_mm: Sequence[float] = ROI_MM,
    px_per_mm: float = PX_PER_MM,
    threshold_value: int = THRESHOLD_VALUE,
    close_kernel_size: int = CLOSE_KERNEL_SIZE,
    close_iterations: int = CLOSE_ITERATIONS,
    erosion_kernel_size: int = EROSION_KERNEL_SIZE,
    erosion_iterations: int = EROSION_ITERATIONS,
    detect_marks: bool = DETECT_MARKS,
    mark_min_area: int = MARK_MIN_AREA,
    mark_max_area: int = MARK_MAX_AREA,
    mark_min_size: int = MARK_MIN_SIZE,
    mark_max_size: int = MARK_MAX_SIZE,
    mark_upper_region_quantile: float = MARK_UPPER_REGION_QUANTILE,
    dot_max_aspect_ratio: float = DOT_MAX_ASPECT_RATIO,
    dot_max_eccentricity: float = DOT_MAX_ECCENTRICITY,
    diaeresis_pair_min_dx: int = DIAERESIS_PAIR_MIN_DX,
    diaeresis_pair_max_dx: int = DIAERESIS_PAIR_MAX_DX,
    diaeresis_pair_max_dy: int = DIAERESIS_PAIR_MAX_DY,
    dot_radius_scale: float = DOT_RADIUS_SCALE,
    dot_spline_points: int = DOT_SPLINE_POINTS,
    accent_min_path_pixels: int = ACCENT_MIN_PATH_PIXELS,
    min_path_pixels: int = MIN_PATH_PIXELS,
    order_strategy: str = ORDER_STRATEGY,
    merge_close_paths: bool = MERGE_CLOSE_PATHS,
    merge_max_gap_pixels: int = MERGE_MAX_GAP_PIXELS,
    merge_max_vertical_gap_pixels: int = MERGE_MAX_VERTICAL_GAP_PIXELS,
    merge_connector_points: int = MERGE_CONNECTOR_POINTS,
    merge_with_retrace: bool = MERGE_WITH_RETRACE,
    retrace_attach_max_distance: int = RETRACE_ATTACH_MAX_DISTANCE,
    retrace_max_backtrack_pixels: int = RETRACE_MAX_BACKTRACK_PIXELS,
    retrace_connector_points: int = RETRACE_CONNECTOR_POINTS,
    global_retrace_enabled: bool = GLOBAL_RETRACE_ENABLED,
    global_retrace_attach_max_distance: int = GLOBAL_RETRACE_ATTACH_MAX_DISTANCE,
    global_retrace_max_route_pixels: int = GLOBAL_RETRACE_MAX_ROUTE_PIXELS,
    global_retrace_connector_points: int = GLOBAL_RETRACE_CONNECTOR_POINTS,
    global_retrace_allow_marks: bool = GLOBAL_RETRACE_ALLOW_MARKS,
    postpone_marks_to_end: bool = POSTPONE_MARKS_TO_END,
    avoid_merging_marks: bool = AVOID_MERGING_MARKS,
    spline_smoothing: float = SPLINE_SMOOTHING,
    spline_points: int = SPLINE_POINTS,
    spline_degree: int = SPLINE_DEGREE,
):
    """
    Ejecuta el pipeline completo V7.

    Retorna:
        splines: lista de diccionarios con puntos en píxeles y metadata.
        spline_arrays_mm: lista de arrays Nx2 en milímetros A4.
        data: diccionario de imágenes, marcas y caminos intermedios.
        graph_info: información del grafo del esqueleto sin marcas.
    """
    image_path = render_text_image(
        text,
        font_path,
        output_dir=output_dir,
        image_size=image_size,
        max_font_size=max_font_size,
        margin=margin,
        roi_mm=roi_mm,
        px_per_mm=px_per_mm,
    )

    data = preprocess_text_image(
        image_path,
        threshold_value=threshold_value,
        close_kernel_size=close_kernel_size,
        close_iterations=close_iterations,
        erosion_kernel_size=erosion_kernel_size,
        erosion_iterations=erosion_iterations,
        detect_marks=detect_marks,
        mark_min_area=mark_min_area,
        mark_max_area=mark_max_area,
        mark_min_size=mark_min_size,
        mark_max_size=mark_max_size,
        mark_upper_region_quantile=mark_upper_region_quantile,
        dot_max_aspect_ratio=dot_max_aspect_ratio,
        dot_max_eccentricity=dot_max_eccentricity,
        diaeresis_pair_min_dx=diaeresis_pair_min_dx,
        diaeresis_pair_max_dx=diaeresis_pair_max_dx,
        diaeresis_pair_max_dy=diaeresis_pair_max_dy,
    )

    mark_paths = mark_components_to_paths(
        data["mark_components"],
        data["connected_labels"],
        dot_num_points=dot_spline_points,
        dot_radius_scale=dot_radius_scale,
        accent_min_path_pixels=accent_min_path_pixels,
    )

    raw_skeleton_paths, graph_info = skeleton_to_paths(
        data["skeleton_no_marks_bool"],
        min_path_pixels=min_path_pixels,
    )

    raw_paths = raw_skeleton_paths + mark_paths

    if postpone_marks_to_end:
        ordered_skeleton_paths = order_paths_for_writing(
            raw_skeleton_paths,
            strategy=order_strategy,
        )
        ordered_mark_paths = order_paths_for_writing(
            mark_paths,
            strategy="left_to_right",
        )
        ordered_paths = ordered_skeleton_paths + ordered_mark_paths
    else:
        ordered_paths = order_paths_for_writing(
            raw_paths,
            strategy=order_strategy,
        )

    if merge_close_paths:
        final_paths = merge_close_ordered_paths(
            ordered_paths,
            max_gap=merge_max_gap_pixels,
            max_vertical_gap=merge_max_vertical_gap_pixels,
            connector_points=merge_connector_points,
            merge_with_retrace=merge_with_retrace,
            retrace_attach_max_distance=retrace_attach_max_distance,
            retrace_max_backtrack_pixels=retrace_max_backtrack_pixels,
            retrace_connector_points=retrace_connector_points,
            global_retrace_enabled=global_retrace_enabled,
            global_retrace_attach_max_distance=global_retrace_attach_max_distance,
            global_retrace_max_route_pixels=global_retrace_max_route_pixels,
            global_retrace_connector_points=global_retrace_connector_points,
            global_retrace_allow_marks=global_retrace_allow_marks,
            avoid_merging_marks=avoid_merging_marks,
        )
    else:
        final_paths = ordered_paths

    splines = paths_to_splines(
        final_paths,
        smoothing=spline_smoothing,
        num_points=spline_points,
        degree=spline_degree,
    )

    spline_arrays_px = [spline["points"] for spline in splines]
    spline_arrays_mm = splines_px_to_a4_mm(splines, px_per_mm=px_per_mm)

    for spline, points_mm in zip(splines, spline_arrays_mm):
        spline["points_mm"] = points_mm

    data.update({
        "image_path": image_path,
        "raw_skeleton_paths": raw_skeleton_paths,
        "mark_paths": mark_paths,
        "raw_paths": raw_paths,
        "ordered_paths": ordered_paths,
        "final_paths": final_paths,
        "spline_arrays_px": spline_arrays_px,
        "spline_arrays_mm": spline_arrays_mm,
        "roi_mm": tuple(float(v) for v in roi_mm),
        "px_per_mm": float(px_per_mm),
    })

    return splines, spline_arrays_mm, data, graph_info


__all__ = [
    "run_full_pipeline",
    "A4_WIDTH_MM",
    "A4_HEIGHT_MM",
    "PX_PER_MM",
    "IMAGE_SIZE",
    "ROI_MM",
    "MARGIN",
    "ROI_MARGIN_MM",
]
