"""
letter_trajectory_functions_v6.py  —  Módulo de funciones V6

Contiene todas las funciones del pipeline de detección y generación de
trayectorias de letras sobre una hoja A4 virtual:

    1. Generación de imagen PIL sobre lienzo A4.
    2. Preprocesamiento: binarización, detección de marcas superiores (puntos
       de i/j, diéresis, tildes/acentos) y skeletonization.
    3. Conversión del esqueleto a grafo y extracción de caminos.
    4. Ordenamiento y fusión de caminos (escritura izquierda a derecha).
    5. Ajuste de splines.
    6. Conversión de píxeles a milímetros sobre la hoja A4.

Nota: esta versión no depende de librerías de graficado.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev


# ===========================================================================
# Parámetros por defecto del pipeline A4
# ===========================================================================

A4_WIDTH_MM  = 210.0
A4_HEIGHT_MM = 297.0
PX_PER_MM    = 6.0

CANVAS_WIDTH_PX  = int(round(A4_WIDTH_MM  * PX_PER_MM))
CANVAS_HEIGHT_PX = int(round(A4_HEIGHT_MM * PX_PER_MM))
IMAGE_SIZE = (CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX)

ROI_X_MM = 0.0
ROI_Y_MM = 249.0
ROI_W_MM = 210.0
ROI_H_MM = 46.0
ROI_MM   = (ROI_X_MM, ROI_Y_MM, ROI_W_MM, ROI_H_MM)

ROI_MARGIN_MM = 2.0
MARGIN = int(round(ROI_MARGIN_MM * PX_PER_MM))


# ===========================================================================
# Sección 3 — Generación de imagen de la palabra
# ===========================================================================

def get_valid_font_path(font_path):
    """
    Retorna la fuente indicada si existe. Si no existe, busca una fuente
    TrueType/OpenType común del sistema sin depender de librerías de graficado.
    """
    if font_path and os.path.exists(font_path):
        return font_path

    preferred_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]

    for candidate in preferred_fonts:
        if os.path.exists(candidate):
            print("Advertencia: no se encontró la fuente indicada.")
            print(f"Usando fuente de respaldo: {candidate}")
            return candidate

    font_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
    ]

    for font_dir in font_dirs:
        if not os.path.isdir(font_dir):
            continue
        for root, _, files in os.walk(font_dir):
            for file_name in files:
                if file_name.lower().endswith((".ttf", ".otf")):
                    fallback = os.path.join(root, file_name)
                    print("Advertencia: no se encontró la fuente indicada.")
                    print(f"Usando fuente de respaldo: {fallback}")
                    return fallback

    raise FileNotFoundError(
        "No se encontró la fuente indicada ni una fuente .ttf/.otf de respaldo. "
        "Instala fonts-dejavu-core o define FONT_PATH con una ruta válida."
    )


def get_next_filename(output_dir, text, font_path):
    """Genera un nombre incremental para no sobrescribir imágenes anteriores."""
    os.makedirs(output_dir, exist_ok=True)

    safe_text  = str(text).replace(" ", "_")
    font_name  = os.path.basename(font_path).split(".")[0]

    i = 1
    while True:
        file_name = f"{safe_text}_{font_name}_{i}.png"
        file_path = os.path.join(output_dir, file_name)
        if not os.path.exists(file_path):
            return file_path
        i += 1


def image_size_to_wh(image_size):
    """
    Permite usar image_size como entero cuadrado o como tupla (width_px, height_px).
    """
    if isinstance(image_size, (tuple, list, np.ndarray)):
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)


def mm_to_px(value_mm, px_per_mm=PX_PER_MM):
    """Convierte milímetros a píxeles usando la misma escala en X e Y."""
    return int(round(float(value_mm) * float(px_per_mm)))


def roi_mm_to_px(roi_mm, px_per_mm=PX_PER_MM):
    """
    Convierte un ROI físico (x_mm, y_mm, w_mm, h_mm) a píxeles.
    """
    x_mm, y_mm, w_mm, h_mm = roi_mm
    return (
        mm_to_px(x_mm, px_per_mm),
        mm_to_px(y_mm, px_per_mm),
        mm_to_px(w_mm, px_per_mm),
        mm_to_px(h_mm, px_per_mm),
    )


def validate_a4_roi(roi_mm, a4_width_mm=A4_WIDTH_MM, a4_height_mm=A4_HEIGHT_MM):
    """Verifica que el recuadro esté dentro de la hoja A4."""
    x_mm, y_mm, w_mm, h_mm = roi_mm

    if x_mm < 0 or y_mm < 0:
        raise ValueError("El ROI no puede empezar fuera de la hoja A4.")
    if w_mm <= 0 or h_mm <= 0:
        raise ValueError("El ROI debe tener ancho y alto positivos.")
    if x_mm + w_mm > a4_width_mm:
        raise ValueError("El ROI se sale del ancho de la hoja A4.")
    if y_mm + h_mm > a4_height_mm:
        raise ValueError("El ROI se sale del alto de la hoja A4.")


def points_px_to_a4_mm(points_xy, px_per_mm=PX_PER_MM):
    """
    Convierte puntos [x_px, y_px] a [x_mm, y_mm] sobre la hoja A4.
    No normaliza a [0, 1] ni estira ejes; solo divide por px/mm.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    return points_xy / float(px_per_mm)


def splines_px_to_a4_mm(splines, px_per_mm=PX_PER_MM):
    """Convierte todos los splines de píxeles a milímetros."""
    return [points_px_to_a4_mm(spline["points"], px_per_mm) for spline in splines]


def render_text_image(
    text,
    font_path,
    output_dir="letras_png_A4",
    image_size=IMAGE_SIZE,
    max_font_size=None,
    margin=0,
    roi_mm=ROI_MM,
    px_per_mm=PX_PER_MM,
):
    """
    Genera una imagen RGB con fondo blanco y texto negro sobre una hoja A4.

    La palabra se ubica dentro del ROI físico indicado en milímetros.
    El recuadro rojo NO se dibuja en la imagen (solo en visualizaciones).
    """
    validate_a4_roi(roi_mm)
    font_path = get_valid_font_path(font_path)
    file_path = get_next_filename(output_dir, text, font_path)

    image_w_px, image_h_px = image_size_to_wh(image_size)
    img  = Image.new("RGB", (image_w_px, image_h_px), "white")
    draw = ImageDraw.Draw(img)

    roi_x_px, roi_y_px, roi_w_px, roi_h_px = roi_mm_to_px(roi_mm, px_per_mm)

    margin_px   = int(round(margin))
    usable_x_px = roi_x_px + margin_px
    usable_y_px = roi_y_px + margin_px
    usable_w_px = max(1, roi_w_px - 2 * margin_px)
    usable_h_px = max(1, roi_h_px - 2 * margin_px)

    font_size = max(12, int(round(usable_h_px * 1.8))) if max_font_size is None else int(max_font_size)

    while font_size > 10:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= usable_w_px and text_h <= usable_h_px:
            break
        font_size -= 1

    font   = ImageFont.truetype(font_path, font_size)
    bbox   = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = usable_x_px + (usable_w_px - text_w) / 2 - bbox[0]
    y = usable_y_px + (usable_h_px - text_h) / 2 - bbox[1]

    draw.text((x, y), text, fill="black", font=font)
    img.save(file_path)

    print(f"Imagen guardada en: {file_path}")
    print(f"Hoja A4: {A4_WIDTH_MM:.1f} mm x {A4_HEIGHT_MM:.1f} mm")
    print(f"Lienzo: {image_w_px} px x {image_h_px} px")
    print(f"Escala: {px_per_mm:.3f} px/mm")
    print(f"Recuadro ROI [mm]: x={roi_mm[0]}, y={roi_mm[1]}, w={roi_mm[2]}, h={roi_mm[3]}")
    print(f"Recuadro ROI [px]: x={roi_x_px}, y={roi_y_px}, w={roi_w_px}, h={roi_h_px}")
    print(f"Margen interno: {margin_px} px = {margin_px / px_per_mm:.2f} mm")
    print(f"Tamaño de fuente usado: {font_size}")
    print(f"BBox del texto [px]: w={text_w}, h={text_h}")

    return file_path


# ===========================================================================
# Sección 4 — Preprocesamiento, marcas superiores y skeletonization
# ===========================================================================

def _component_pca_features(component_mask):
    """
    Calcula rasgos geométricos simples de un componente conectado.
    Retorna excentricidad, ángulo principal y eigenvalues de PCA.
    """
    ys, xs = np.where(component_mask > 0)
    if len(xs) < 3:
        return {"eccentricity": 0.0, "angle_deg": 0.0, "lambda_major": 0.0, "lambda_minor": 0.0}

    points  = np.column_stack((xs, ys)).astype(float)
    points -= np.mean(points, axis=0)

    cov           = np.cov(points, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order    = np.argsort(eigvals)[::-1]
    eigvals  = eigvals[order]
    eigvecs  = eigvecs[:, order]

    lambda_major = float(max(eigvals[0], 1e-9))
    lambda_minor = float(max(eigvals[1], 1e-9))
    eccentricity = float(np.sqrt(max(0.0, 1.0 - lambda_minor / lambda_major)))
    angle_deg    = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))

    return {"eccentricity": eccentricity, "angle_deg": angle_deg,
            "lambda_major": lambda_major, "lambda_minor": lambda_minor}


def _is_diaeresis_pair(comp_a, comp_b, min_dx=8, max_dx=58, max_dy=18):
    """
    Decide si dos componentes compactos pueden formar una diéresis.
    """
    cxa, cya = comp_a["centroid_xy"]
    cxb, cyb = comp_b["centroid_xy"]

    dx = abs(cxb - cxa)
    dy = abs(cyb - cya)

    if dx < min_dx or dx > max_dx:
        return False
    if dy > max_dy:
        return False

    _, _, wa, ha = comp_a["bbox"]
    _, _, wb, hb = comp_b["bbox"]

    if max(wa, wb) / max(min(wa, wb), 1) > 2.0:
        return False
    if max(ha, hb) / max(min(ha, hb), 1) > 2.0:
        return False

    return True


def detect_upper_mark_components(
    binary_clean,
    min_area=20,
    max_area=1200,
    min_size=4,
    max_size=75,
    upper_region_quantile=0.72,
    dot_max_aspect_ratio=1.65,
    dot_max_eccentricity=0.78,
    diaeresis_pair_min_dx=8,
    diaeresis_pair_max_dx=58,
    diaeresis_pair_max_dy=18,
):
    """
    Detecta marcas superiores separadas de la palabra:
        - dot_i_j: punto compacto de i/j.
        - diaeresis_dot: puntos dobles de diéresis.
        - accent_or_tilde: tilde, acento agudo u otra marca alargada.
    """
    binary_uint8 = (binary_clean > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_uint8, connectivity=8
    )

    foreground_y = np.where(binary_uint8 > 0)[0]
    if len(foreground_y) == 0:
        return [], labels

    y_limit    = np.quantile(foreground_y, upper_region_quantile)
    candidates = []

    for label in range(1, num_labels):
        x    = int(stats[label, cv2.CC_STAT_LEFT])
        y    = int(stats[label, cv2.CC_STAT_TOP])
        w    = int(stats[label, cv2.CC_STAT_WIDTH])
        h    = int(stats[label, cv2.CC_STAT_HEIGHT])
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

        aspect_ratio    = max(w / max(h, 1), h / max(w, 1))
        component_mask  = labels[y:y + h, x:x + w] == label
        pca             = _component_pca_features(component_mask)

        is_compact_dot = (
            aspect_ratio <= dot_max_aspect_ratio
            and pca["eccentricity"] <= dot_max_eccentricity
        )

        candidates.append({
            "label": label,
            "bbox": (x, y, w, h),
            "area": area,
            "centroid_xy": np.array([cx, cy], dtype=float),
            "aspect_ratio": float(aspect_ratio),
            "eccentricity": pca["eccentricity"],
            "angle_deg": pca["angle_deg"],
            "is_compact_dot_candidate": bool(is_compact_dot),
            "mark_kind": "dot_i_j" if is_compact_dot else "accent_or_tilde",
        })

    # Agrupar pares de puntos compactos como diéresis
    compact_indices = [i for i, c in enumerate(candidates) if c["is_compact_dot_candidate"]]
    used = set()

    for i in compact_indices:
        if i in used:
            continue
        best_j, best_dx = None, np.inf
        for j in compact_indices:
            if j <= i or j in used:
                continue
            if _is_diaeresis_pair(
                candidates[i], candidates[j],
                min_dx=diaeresis_pair_min_dx,
                max_dx=diaeresis_pair_max_dx,
                max_dy=diaeresis_pair_max_dy,
            ):
                dx = abs(candidates[j]["centroid_xy"][0] - candidates[i]["centroid_xy"][0])
                if dx < best_dx:
                    best_dx, best_j = dx, j

        if best_j is not None:
            pair_id = len(used) + 1
            candidates[i]["mark_kind"]       = "diaeresis_dot"
            candidates[best_j]["mark_kind"]  = "diaeresis_dot"
            candidates[i]["diaeresis_pair_id"]      = pair_id
            candidates[best_j]["diaeresis_pair_id"] = pair_id
            used.add(i)
            used.add(best_j)

    candidates.sort(key=lambda c: (c["centroid_xy"][0], c["centroid_xy"][1]))
    return candidates, labels


def remove_mark_components_from_skeleton(skeleton_bool, mark_components, padding=2):
    """
    Elimina del esqueleto las zonas donde se detectaron marcas superiores.
    """
    skeleton_no_marks = skeleton_bool.copy()
    height, width = skeleton_no_marks.shape

    for comp in mark_components:
        x, y, w, h = comp["bbox"]
        x0 = max(x - padding, 0)
        y0 = max(y - padding, 0)
        x1 = min(x + w + padding, width)
        y1 = min(y + h + padding, height)
        skeleton_no_marks[y0:y1, x0:x1] = False

    return skeleton_no_marks


def _ellipse_path_from_component(comp, num_points=70, radius_scale=0.85):
    """Crea una trayectoria cerrada tipo elipse para puntos de i/j o diéresis."""
    x, y, w, h = comp["bbox"]
    cx, cy = comp["centroid_xy"]

    rx    = max((w * 0.5) * radius_scale, 2.0)
    ry    = max((h * 0.5) * radius_scale, 2.0)
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
    xs    = cx + rx * np.cos(theta)
    ys    = cy + ry * np.sin(theta)

    return np.column_stack((ys, xs)).astype(float)


def _pca_line_path_from_component(comp, labels, num_points=20):
    """
    Respaldo para marcas muy pequeñas: aproxima el componente con una línea
    siguiendo su eje principal.
    """
    label = comp["label"]
    x, y, w, h = comp["bbox"]

    local    = labels[y:y + h, x:x + w] == label
    ys, xs   = np.where(local > 0)
    if len(xs) == 0:
        cx, cy = comp["centroid_xy"]
        return np.array([[cy, cx]], dtype=float)

    points   = np.column_stack((xs + x, ys + y)).astype(float)
    if len(points) < 2:
        cx, cy = comp["centroid_xy"]
        return np.array([[cy, cx]], dtype=float)

    mean      = np.mean(points, axis=0)
    centered  = points - mean
    cov       = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]

    projections = centered @ direction
    p0       = mean + projections.min() * direction
    p1       = mean + projections.max() * direction
    line_xy  = np.linspace(p0, p1, num_points)

    return np.column_stack((line_xy[:, 1], line_xy[:, 0])).astype(float)


def _accent_path_from_component(comp, labels, min_path_pixels=4):
    """
    Conserva la forma real de una tilde/acento a partir del esqueleto local.
    Si el esqueleto queda demasiado corto, usa PCA como respaldo.
    """
    label = comp["label"]
    x, y, w, h = comp["bbox"]

    local_mask     = labels[y:y + h, x:x + w] == label
    local_skeleton = skeletonize(local_mask > 0)

    local_paths, _ = skeleton_to_paths(local_skeleton, min_path_pixels=min_path_pixels)

    if len(local_paths) == 0:
        pixels = _pca_line_path_from_component(comp, labels)
        return [pixels]

    paths = []
    for path in local_paths:
        local_pixels = path["pixels"].astype(float)
        local_pixels[:, 0] += y
        local_pixels[:, 1] += x
        paths.append(local_pixels)

    return paths


def mark_components_to_paths(
    mark_components,
    labels,
    dot_num_points=70,
    dot_radius_scale=0.85,
    accent_min_path_pixels=4,
):
    """
    Convierte marcas superiores detectadas en caminos:
        - puntos i/j y diéresis → elipse cerrada.
        - tilde/acento           → esqueleto real del componente.
    """
    mark_paths = []

    for idx, comp in enumerate(mark_components):
        mark_kind = comp.get("mark_kind", "unknown")

        if mark_kind in ["dot_i_j", "diaeresis_dot"]:
            pixels = _ellipse_path_from_component(
                comp, num_points=dot_num_points, radius_scale=dot_radius_scale
            )
            mark_paths.append({
                "pixels": pixels,
                "closed": True,
                "is_mark": True,
                "is_dot": mark_kind == "dot_i_j",
                "is_diaeresis": mark_kind == "diaeresis_dot",
                "is_accent": False,
                "mark_kind": mark_kind,
                "source": mark_kind,
                "mark_id": idx + 1,
                "component": comp,
            })
        else:
            accent_paths = _accent_path_from_component(
                comp, labels, min_path_pixels=accent_min_path_pixels
            )
            for k, pixels in enumerate(accent_paths):
                mark_paths.append({
                    "pixels": pixels,
                    "closed": False,
                    "is_mark": True,
                    "is_dot": False,
                    "is_diaeresis": False,
                    "is_accent": True,
                    "mark_kind": "accent_or_tilde",
                    "source": "accent_or_tilde",
                    "mark_id": idx + 1,
                    "accent_part": k + 1,
                    "component": comp,
                })

    return mark_paths


def preprocess_text_image(
    image_path,
    threshold_value=200,
    close_kernel_size=5,
    close_iterations=1,
    erosion_kernel_size=3,
    erosion_iterations=0,
    detect_marks=True,
    mark_min_area=20,
    mark_max_area=1200,
    mark_min_size=4,
    mark_max_size=75,
    mark_upper_region_quantile=0.72,
    dot_max_aspect_ratio=1.65,
    dot_max_eccentricity=0.78,
    diaeresis_pair_min_dx=8,
    diaeresis_pair_max_dx=58,
    diaeresis_pair_max_dy=18,
):
    """
    Lee la imagen, la binariza, detecta marcas superiores y obtiene el esqueleto.

    Retorna un diccionario con:
        image_rgb, gray, binary_clean, skeleton_bool, skeleton_no_marks_bool,
        skeleton_uint8, skeleton_overlay, mark_components, connected_labels.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)

    if erosion_iterations > 0:
        erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        binary_clean   = cv2.erode(binary_clean, erosion_kernel, iterations=erosion_iterations)

    mark_components  = []
    connected_labels = None

    if detect_marks:
        mark_components, connected_labels = detect_upper_mark_components(
            binary_clean,
            min_area=mark_min_area,
            max_area=mark_max_area,
            min_size=mark_min_size,
            max_size=mark_max_size,
            upper_region_quantile=mark_upper_region_quantile,
            dot_max_aspect_ratio=dot_max_aspect_ratio,
            dot_max_eccentricity=dot_max_eccentricity,
            diaeresis_pair_min_dx=diaeresis_pair_min_dx,
            diaeresis_pair_max_dx=diaeresis_pair_max_dx,
            diaeresis_pair_max_dy=diaeresis_pair_max_dy,
        )
    else:
        connected_labels = cv2.connectedComponentsWithStats(
            (binary_clean > 0).astype(np.uint8), connectivity=8
        )[1]

    skeleton_bool         = skeletonize(binary_clean > 0)
    skeleton_no_marks_bool = remove_mark_components_from_skeleton(skeleton_bool, mark_components)
    skeleton_uint8        = (skeleton_bool * 255).astype(np.uint8)

    skeleton_overlay            = img_rgb.copy()
    y_coords, x_coords          = np.where(skeleton_bool)
    skeleton_overlay[y_coords, x_coords] = [255, 0, 0]

    return {
        "image_rgb": img_rgb,
        "gray": gray,
        "binary_clean": binary_clean,
        "skeleton_bool": skeleton_bool,
        "skeleton_no_marks_bool": skeleton_no_marks_bool,
        "skeleton_uint8": skeleton_uint8,
        "skeleton_overlay": skeleton_overlay,
        "mark_components": mark_components,
        "connected_labels": connected_labels,
    }


# ===========================================================================
# Sección 5 — Esqueleto como grafo
# ===========================================================================

NEIGHBOR_OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


def get_skeleton_pixels(skeleton_bool):
    """Retorna los píxeles activos del esqueleto como set de tuplas (y, x)."""
    coords = np.column_stack(np.where(skeleton_bool > 0))
    return set(map(tuple, coords))


def get_pixel_neighbors(pixel, skeleton_pixels):
    """Obtiene vecinos 8-conectados de un píxel del esqueleto."""
    y, x = pixel
    return [
        (y + dy, x + dx)
        for dy, dx in NEIGHBOR_OFFSETS_8
        if (y + dy, x + dx) in skeleton_pixels
    ]


def classify_skeleton_pixels(skeleton_bool):
    """
    Clasifica los píxeles del esqueleto en endpoints, junctions y normal_points.
    También retorna degree_map.
    """
    skeleton_pixels = get_skeleton_pixels(skeleton_bool)

    endpoints, junctions, normal_points, isolated_points = [], [], [], []
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

    return {
        "skeleton_pixels": skeleton_pixels,
        "degree_map": degree_map,
        "endpoints": np.array(endpoints, dtype=np.int32),
        "junctions": np.array(junctions, dtype=np.int32),
        "normal_points": np.array(normal_points, dtype=np.int32),
        "isolated_points": np.array(isolated_points, dtype=np.int32),
    }


# ===========================================================================
# Sección 6 — Caminos continuos
# ===========================================================================

def edge_key(pixel_a, pixel_b):
    """Identificador único de una arista entre dos píxeles."""
    return tuple(sorted((pixel_a, pixel_b)))


def trace_path_from_node(start_pixel, next_pixel, skeleton_pixels, degree_map, graph_nodes, visited_edges):
    """
    Traza un camino desde un nodo del grafo hasta otro nodo.
    """
    path = [start_pixel]
    previous_pixel = start_pixel
    current_pixel  = next_pixel

    visited_edges.add(edge_key(start_pixel, next_pixel))
    max_steps = len(skeleton_pixels) + 10

    for _ in range(max_steps):
        path.append(current_pixel)
        if current_pixel in graph_nodes and current_pixel != start_pixel:
            break

        neighbors  = get_pixel_neighbors(current_pixel, skeleton_pixels)
        candidates = [
            n for n in neighbors
            if n != previous_pixel and edge_key(current_pixel, n) not in visited_edges
        ]
        if not candidates:
            break

        next_candidate  = candidates[0]
        previous_pixel, current_pixel = current_pixel, next_candidate
        visited_edges.add(edge_key(previous_pixel, current_pixel))

    return np.array(path, dtype=np.int32)


def trace_closed_path(start_pixel, next_pixel, skeleton_pixels, visited_edges):
    """
    Traza un ciclo cerrado cuando el componente no tiene extremos (letras como 'o', 'O').
    """
    path = [start_pixel]
    previous_pixel = start_pixel
    current_pixel  = next_pixel

    visited_edges.add(edge_key(start_pixel, next_pixel))
    max_steps = len(skeleton_pixels) + 10

    for _ in range(max_steps):
        path.append(current_pixel)

        if current_pixel == start_pixel and len(path) > 2:
            break

        neighbors           = get_pixel_neighbors(current_pixel, skeleton_pixels)
        candidates          = [n for n in neighbors if n != previous_pixel]
        unvisited_candidates = [n for n in candidates if edge_key(current_pixel, n) not in visited_edges]

        if start_pixel in candidates and len(path) > 3:
            next_candidate = start_pixel
        elif unvisited_candidates:
            next_candidate = unvisited_candidates[0]
        else:
            break

        if edge_key(current_pixel, next_candidate) not in visited_edges:
            visited_edges.add(edge_key(current_pixel, next_candidate))

        previous_pixel, current_pixel = current_pixel, next_candidate

    return np.array(path, dtype=np.int32)


def skeleton_to_paths(skeleton_bool, min_path_pixels=12):
    """
    Convierte el esqueleto en una lista de caminos ordenados.

    Retorna (paths, graph_info) donde paths es una lista de dicts:
        {"pixels": arreglo Nx2 [y, x], "closed": bool}
    """
    graph_info      = classify_skeleton_pixels(skeleton_bool)
    skeleton_pixels = graph_info["skeleton_pixels"]
    degree_map      = graph_info["degree_map"]

    graph_nodes   = {p for p, d in degree_map.items() if d != 2}
    visited_edges = set()
    paths         = []

    for node in sorted(graph_nodes):
        for neighbor in get_pixel_neighbors(node, skeleton_pixels):
            if edge_key(node, neighbor) in visited_edges:
                continue
            path_pixels = trace_path_from_node(
                node, neighbor, skeleton_pixels, degree_map, graph_nodes, visited_edges
            )
            if len(path_pixels) >= min_path_pixels:
                paths.append({"pixels": path_pixels, "closed": False})

    for pixel in sorted(skeleton_pixels):
        for neighbor in get_pixel_neighbors(pixel, skeleton_pixels):
            if edge_key(pixel, neighbor) in visited_edges:
                continue
            path_pixels = trace_closed_path(pixel, neighbor, skeleton_pixels, visited_edges)
            if len(path_pixels) >= min_path_pixels:
                paths.append({"pixels": path_pixels, "closed": True})

    return paths, graph_info


# ===========================================================================
# Sección 7 — Ordenar y fusionar caminos
# ===========================================================================

def pixels_to_xy(path_pixels):
    """Convierte puntos de imagen [y, x] a puntos cartesianos [x, y]."""
    return np.column_stack((path_pixels[:, 1], path_pixels[:, 0])).astype(float)


def path_start_xy(path):
    return pixels_to_xy(path["pixels"])[0]


def path_end_xy(path):
    return pixels_to_xy(path["pixels"])[-1]


def path_bbox_xy(path):
    xy = pixels_to_xy(path["pixels"])
    return {
        "min_x":    float(np.min(xy[:, 0])),
        "max_x":    float(np.max(xy[:, 0])),
        "min_y":    float(np.min(xy[:, 1])),
        "max_y":    float(np.max(xy[:, 1])),
        "center_x": float(np.mean(xy[:, 0])),
        "center_y": float(np.mean(xy[:, 1])),
    }


def reverse_path(path):
    """Invierte el sentido de un camino preservando su metadata."""
    new_path = path.copy()
    new_path["pixels"] = path["pixels"][::-1].copy()
    return new_path


def rotate_closed_path_to_nearest_point(path, reference_xy):
    """Para caminos cerrados, cambia el punto de inicio al más cercano a una referencia."""
    pixels    = path["pixels"]
    xy        = pixels_to_xy(pixels)
    idx       = int(np.argmin(np.linalg.norm(xy - reference_xy, axis=1)))
    new_path  = path.copy()
    new_path["pixels"] = np.roll(pixels, -idx, axis=0)
    return new_path


def orient_path_near_reference(path, reference_xy=None):
    """Orienta el camino hacia el extremo más cercano a la referencia."""
    xy = pixels_to_xy(path["pixels"])

    if path.get("closed", False) and reference_xy is not None:
        return rotate_closed_path_to_nearest_point(path, reference_xy)

    if reference_xy is None:
        return reverse_path(path) if xy[-1, 0] < xy[0, 0] else path

    start_dist = np.linalg.norm(xy[0] - reference_xy)
    end_dist   = np.linalg.norm(xy[-1] - reference_xy)
    return reverse_path(path) if end_dist < start_dist else path


def order_paths_nearest_neighbor(paths):
    """Ordena caminos con heurística de vecino más cercano."""
    if not paths:
        return []

    remaining = [p.copy() for p in paths]
    min_x_values = [np.min(pixels_to_xy(p["pixels"])[:, 0]) for p in remaining]
    first_idx    = int(np.argmin(min_x_values))
    current_path = orient_path_near_reference(remaining.pop(first_idx))
    ordered      = [current_path]
    current_end  = path_end_xy(current_path)

    while remaining:
        best_idx, best_cost = None, np.inf
        for idx, path in enumerate(remaining):
            xy = pixels_to_xy(path["pixels"])
            if path.get("closed", False):
                cost = np.min(np.linalg.norm(xy - current_end, axis=1))
            else:
                cost = min(np.linalg.norm(xy[0] - current_end), np.linalg.norm(xy[-1] - current_end))
            if cost < best_cost:
                best_cost, best_idx = cost, idx

        next_path = orient_path_near_reference(remaining.pop(best_idx), current_end)
        ordered.append(next_path)
        current_end = path_end_xy(next_path)

    return ordered


def _path_sort_priority(path):
    """Prioridad para ordenar marcas después del cuerpo de la letra."""
    if not path.get("is_mark", False):
        return 0
    kind = path.get("mark_kind", "")
    return {"accent_or_tilde": 1, "diaeresis_dot": 2, "dot_i_j": 3}.get(kind, 4)


def order_paths_left_to_right(paths):
    """Ordena caminos priorizando escritura de izquierda a derecha."""
    if not paths:
        return []

    paths_with_bbox = [(p, path_bbox_xy(p)) for p in paths]

    def sort_key(item):
        path, bbox = item
        is_mark   = path.get("is_mark", False)
        is_closed = path.get("closed", False)
        x_key = bbox["center_x"] if (is_mark or is_closed) else bbox["min_x"]
        return (x_key, _path_sort_priority(path), bbox["center_y"])

    sorted_items = sorted(paths_with_bbox, key=sort_key)

    ordered, previous_end = [], None
    for path, _ in sorted_items:
        oriented = orient_path_near_reference(path, previous_end)
        ordered.append(oriented)
        previous_end = path_end_xy(oriented)

    return ordered


def order_paths_for_writing(paths, strategy="left_to_right"):
    """Selecciona la estrategia de ordenamiento de caminos."""
    if strategy == "nearest":
        return order_paths_nearest_neighbor(paths)
    if strategy == "left_to_right":
        return order_paths_left_to_right(paths)
    raise ValueError("strategy debe ser 'left_to_right' o 'nearest'")


def add_linear_connector(path_a, path_b, num_points=8):
    """Une dos caminos con puntos lineales entre el final de A y el inicio de B."""
    end_a   = path_a["pixels"][-1]
    start_b = path_b["pixels"][0]
    connector = np.linspace(end_a, start_b, num_points + 2)[1:-1] if num_points > 0 else np.empty((0, 2), dtype=float)

    new_path = path_a.copy()
    new_path["pixels"]       = np.vstack([path_a["pixels"], connector, path_b["pixels"]])
    new_path["closed"]       = False
    new_path["is_mark"]      = False
    new_path["is_dot"]       = False
    new_path["is_diaeresis"] = False
    new_path["is_accent"]    = False
    new_path["mark_kind"]    = None
    new_path["source"]       = "merged_path"
    new_path["merged_count"] = path_a.get("merged_count", 1) + path_b.get("merged_count", 1)
    new_path["retrace_count"] = path_a.get("retrace_count", 0) + path_b.get("retrace_count", 0)
    return new_path


def can_merge_paths(path_a, path_b, max_gap=30, max_vertical_gap=50):
    """Decide si dos caminos se pueden fusionar directamente."""
    if path_a.get("is_mark", False) or path_b.get("is_mark", False):
        return False
    if path_a.get("closed", False) or path_b.get("closed", False):
        return False

    end_a   = path_end_xy(path_a)
    start_b = path_start_xy(path_b)
    distance     = np.linalg.norm(start_b - end_a)
    vertical_gap = abs(start_b[1] - end_a[1])

    return distance <= max_gap and vertical_gap <= max_vertical_gap


def _path_cumulative_length_pixels(path_pixels):
    """Longitud acumulada de un camino en píxeles."""
    if len(path_pixels) <= 1:
        return np.array([0.0])
    diffs      = np.diff(path_pixels.astype(float), axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.insert(np.cumsum(seg_lengths), 0, 0.0)


def find_retrace_candidate(current_path, next_path, attach_max_distance=16, max_backtrack_pixels=75):
    """
    Busca si next_path puede unirse a un punto anterior de current_path
    (simula retroceso del lápiz al estilo escritura humana).
    """
    if current_path.get("is_mark", False) or next_path.get("is_mark", False):
        return None
    if current_path.get("closed", False) or next_path.get("closed", False):
        return None

    current_pixels = current_path["pixels"]
    current_xy     = pixels_to_xy(current_pixels)
    cumulative     = _path_cumulative_length_pixels(current_pixels)
    total_length   = cumulative[-1]

    best = None
    for candidate in [next_path, reverse_path(next_path)]:
        candidate_start  = path_start_xy(candidate)
        distances        = np.linalg.norm(current_xy - candidate_start, axis=1)
        attach_idx       = int(np.argmin(distances))
        attach_distance  = float(distances[attach_idx])
        backtrack_length = float(total_length - cumulative[attach_idx])

        if attach_idx >= len(current_pixels) - 3:
            continue
        if attach_distance > attach_max_distance:
            continue
        if backtrack_length > max_backtrack_pixels:
            continue

        cost = attach_distance + 0.15 * backtrack_length
        if best is None or cost < best["cost"]:
            best = {
                "path": candidate,
                "attach_idx": attach_idx,
                "attach_distance": attach_distance,
                "backtrack_length": backtrack_length,
                "cost": cost,
            }

    return best


def add_retrace_connector(current_path, next_path, attach_idx, connector_points=4):
    """
    Fusiona current_path con next_path regresando sobre el mismo camino.
    """
    current_pixels  = current_path["pixels"]
    next_pixels     = next_path["pixels"]
    retrace_segment = current_pixels[attach_idx:-1][::-1]
    attach_point    = current_pixels[attach_idx]
    next_start      = next_pixels[0]

    connector = (
        np.linspace(attach_point, next_start, connector_points + 2)[1:-1]
        if connector_points > 0
        else np.empty((0, 2), dtype=float)
    )

    new_path = current_path.copy()
    new_path["pixels"]        = np.vstack([current_pixels, retrace_segment, connector, next_pixels])
    new_path["closed"]        = False
    new_path["is_mark"]       = False
    new_path["is_dot"]        = False
    new_path["is_diaeresis"]  = False
    new_path["is_accent"]     = False
    new_path["mark_kind"]     = None
    new_path["source"]        = "merged_retrace_path"
    new_path["merged_count"]  = current_path.get("merged_count", 1) + next_path.get("merged_count", 1)
    new_path["retrace_count"] = current_path.get("retrace_count", 0) + next_path.get("retrace_count", 0) + 1
    return new_path


def merge_close_ordered_paths(
    ordered_paths,
    max_gap=30,
    max_vertical_gap=50,
    connector_points=8,
    merge_with_retrace=True,
    retrace_attach_max_distance=16,
    retrace_max_backtrack_pixels=75,
    retrace_connector_points=4,
):
    """Fusiona caminos consecutivos si están suficientemente cerca."""
    if not ordered_paths:
        return []

    merged_paths = []
    current = ordered_paths[0]

    for next_path in ordered_paths[1:]:
        next_path = orient_path_near_reference(next_path, path_end_xy(current))

        if can_merge_paths(current, next_path, max_gap=max_gap, max_vertical_gap=max_vertical_gap):
            current = add_linear_connector(current, next_path, num_points=connector_points)
            continue

        if merge_with_retrace:
            candidate = find_retrace_candidate(
                current, next_path,
                attach_max_distance=retrace_attach_max_distance,
                max_backtrack_pixels=retrace_max_backtrack_pixels,
            )
            if candidate is not None:
                current = add_retrace_connector(
                    current, candidate["path"],
                    attach_idx=candidate["attach_idx"],
                    connector_points=retrace_connector_points,
                )
                continue

        merged_paths.append(current)
        current = next_path

    merged_paths.append(current)
    return merged_paths


# ===========================================================================
# Sección 8 — Ajuste de splines
# ===========================================================================

def remove_consecutive_duplicates(points_xy):
    """Elimina puntos repetidos consecutivos."""
    if len(points_xy) <= 1:
        return points_xy
    keep     = np.ones(len(points_xy), dtype=bool)
    keep[1:] = np.any(np.diff(points_xy, axis=0) != 0, axis=1)
    return points_xy[keep]


def linear_resample(points_xy, num_points=120):
    """Re-muestreo lineal de respaldo."""
    points_xy = remove_consecutive_duplicates(points_xy)
    if len(points_xy) <= 1:
        return points_xy

    seg_lengths = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    distance    = np.insert(np.cumsum(seg_lengths), 0, 0)

    if distance[-1] == 0:
        return points_xy

    new_distance = np.linspace(0, distance[-1], num_points)
    x_new = np.interp(new_distance, distance, points_xy[:, 0])
    y_new = np.interp(new_distance, distance, points_xy[:, 1])
    return np.column_stack((x_new, y_new))


def fit_spline_to_path(points_xy, smoothing=2.0, num_points=120, degree=3, closed=False):
    """
    Ajusta un spline 2D a un camino.
    """
    points_xy = remove_consecutive_duplicates(points_xy)
    if len(points_xy) < 2:
        return points_xy

    if closed and np.linalg.norm(points_xy[0] - points_xy[-1]) > 1e-6:
        points_xy = np.vstack([points_xy, points_xy[0]])

    k = min(degree, len(points_xy) - 1)
    if k < 2:
        return linear_resample(points_xy, num_points=num_points)

    try:
        tck, _ = splprep([points_xy[:, 0], points_xy[:, 1]], s=smoothing, k=k, per=closed)
        u_new  = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack((x_new, y_new))
    except Exception as error:
        print(f"No se pudo ajustar spline. Usando interpolación lineal. Error: {error}")
        return linear_resample(points_xy, num_points=num_points)


def paths_to_splines(ordered_paths, smoothing=2.0, num_points=120, degree=3):
    """
    Convierte una lista de caminos ordenados en una lista de splines.
    Preserva metadata de marcas, fusiones y retrocesos.
    """
    splines = []
    for idx, path in enumerate(ordered_paths):
        control_points_xy = pixels_to_xy(path["pixels"])
        spline_points_xy  = fit_spline_to_path(
            control_points_xy,
            smoothing=smoothing,
            num_points=num_points,
            degree=degree,
            closed=path.get("closed", False),
        )
        splines.append({
            "id":               idx + 1,
            "control_points":   control_points_xy,
            "points":           spline_points_xy,
            "closed":           path.get("closed", False),
            "is_mark":          path.get("is_mark", False),
            "is_dot":           path.get("is_dot", False),
            "is_diaeresis":     path.get("is_diaeresis", False),
            "is_accent":        path.get("is_accent", False),
            "mark_kind":        path.get("mark_kind", None),
            "source":           path.get("source", "skeleton_path"),
            "merged_count":     path.get("merged_count", 1),
            "retrace_count":    path.get("retrace_count", 0),
        })
    return splines


# ===========================================================================
# Sección 9 — Utilidades sin graficado
# ===========================================================================

def mark_symbol(mark_kind):
    """Símbolo corto para identificar el tipo de marca en logs o depuración."""
    return {"dot_i_j": "·", "diaeresis_dot": "¨", "accent_or_tilde": "~"}.get(mark_kind, "?")


def path_label(idx, item):
    """Etiqueta corta para caminos y splines."""
    label = f"{idx + 1}"
    if item.get("mark_kind"):
        label += mark_symbol(item.get("mark_kind"))
    return label


def _stroke_linewidth(item):
    """Valor conservado por compatibilidad con versiones previas."""
    if item.get("is_dot", False) or item.get("is_diaeresis", False):
        return 4
    if item.get("is_accent", False):
        return 3
    return 3


def _plotting_removed(*args, **kwargs):
    """
    Función de compatibilidad: las gráficas fueron retiradas para que el
    pipeline pueda ejecutarse en entornos ROS sin dependencias de graficado.
    """
    raise RuntimeError(
        "Las funciones de plot fueron retiradas en esta versión. "
        "Usa RViz para visualizar la trayectoria o guarda los puntos generados."
    )


# Alias conservados para evitar errores de importación en código antiguo.
draw_roi_on_axis = _plotting_removed
plot_preprocessing_results = _plotting_removed
plot_detected_marks = _plotting_removed
plot_detected_dots = _plotting_removed
plot_skeleton_graph_points = _plotting_removed
plot_a4_layout = _plotting_removed
plot_paths_on_word = _plotting_removed
plot_splines_on_word = _plotting_removed
plot_spline_only = _plotting_removed
plot_splines_single_line_view = _plotting_removed
plot_splines_a4_mm = _plotting_removed


# ===========================================================================
# Pipeline completo (función de alto nivel)
# ===========================================================================

def run_full_pipeline(
    text,
    font_path,
    output_dir="letras_png_A4",
    image_size=IMAGE_SIZE,
    max_font_size=None,
    margin=MARGIN,
    roi_mm=ROI_MM,
    px_per_mm=PX_PER_MM,
    threshold_value=200,
    close_kernel_size=5,
    close_iterations=1,
    erosion_kernel_size=3,
    erosion_iterations=0,
    detect_marks=True,
    mark_min_area=20,
    mark_max_area=1200,
    mark_min_size=4,
    mark_max_size=75,
    mark_upper_region_quantile=0.72,
    dot_max_aspect_ratio=1.65,
    dot_max_eccentricity=0.78,
    diaeresis_pair_min_dx=8,
    diaeresis_pair_max_dx=58,
    diaeresis_pair_max_dy=18,
    dot_max_aspect_ratio_mark=1.65,
    dot_radius_scale=0.85,
    dot_spline_points=70,
    accent_min_path_pixels=4,
    min_path_pixels=12,
    order_strategy="left_to_right",
    merge_close_paths=True,
    merge_max_gap_pixels=30,
    merge_max_vertical_gap_pixels=50,
    merge_connector_points=8,
    merge_with_retrace=True,
    retrace_attach_max_distance=16,
    retrace_max_backtrack_pixels=75,
    retrace_connector_points=4,
    spline_smoothing=2.0,
    spline_points=150,
    spline_degree=3,
):
    """
    Ejecuta el pipeline completo de imagen → splines.

    Retorna (splines, spline_arrays_mm, data, graph_info) donde:
        splines          — lista de dicts con puntos en px y mm.
        spline_arrays_mm — lista de arrays Nx2 en mm.
        data             — dict con resultados del preprocesamiento.
        graph_info       — dict con info del grafo del esqueleto.
    """
    # 1. Generar imagen
    image_path = render_text_image(
        text, font_path,
        output_dir=output_dir, image_size=image_size,
        max_font_size=max_font_size, margin=margin,
        roi_mm=roi_mm, px_per_mm=px_per_mm,
    )

    # 2. Preprocesar
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

    # 3. Marcas superiores → caminos
    mark_paths = mark_components_to_paths(
        data["mark_components"], data["connected_labels"],
        dot_num_points=dot_spline_points,
        dot_radius_scale=dot_radius_scale,
        accent_min_path_pixels=accent_min_path_pixels,
    )

    # 4. Esqueleto sin marcas → caminos
    raw_skeleton_paths, graph_info = skeleton_to_paths(
        data["skeleton_no_marks_bool"], min_path_pixels=min_path_pixels
    )

    # 5. Unir y ordenar
    raw_paths     = raw_skeleton_paths + mark_paths
    ordered_paths = order_paths_for_writing(raw_paths, strategy=order_strategy)

    # 6. Fusionar
    final_paths = (
        merge_close_ordered_paths(
            ordered_paths,
            max_gap=merge_max_gap_pixels,
            max_vertical_gap=merge_max_vertical_gap_pixels,
            connector_points=merge_connector_points,
            merge_with_retrace=merge_with_retrace,
            retrace_attach_max_distance=retrace_attach_max_distance,
            retrace_max_backtrack_pixels=retrace_max_backtrack_pixels,
            retrace_connector_points=retrace_connector_points,
        )
        if merge_close_paths
        else ordered_paths
    )

    # 7. Splines
    splines = paths_to_splines(
        final_paths,
        smoothing=spline_smoothing,
        num_points=spline_points,
        degree=spline_degree,
    )

    # 8. Conversión a mm
    spline_arrays_mm = splines_px_to_a4_mm(splines, px_per_mm=px_per_mm)
    for spline, points_mm in zip(splines, spline_arrays_mm):
        spline["points_mm"] = points_mm

    return splines, spline_arrays_mm, data, graph_info
