#!/usr/bin/env python3
"""
fourier_image_functions.py — imagen (dibujo animado, line art) -> trazos de
plumón representados con series de Fourier.

Con un solo plumón, lo que hace que el dibujo "se parezca" es reproducir las
líneas, no los colores. La imagen se separa en tres tipos de elementos:

1. LÍNEAS: todo lo fino y más oscuro que su entorno (contornos negros, el
   contorno oliva de Bob Esponja, la nariz, las rayas de las medias...). Se
   detectan con un black-hat morfológico, se esqueletizan, se refinan (centrado
   sub-píxel + extremos, ver fourier_trajectory_functions.refinar_caminos) y el
   plumón recorre su línea central.
2. ZONAS NEGRAS: más gruesas que el plumón (pupilas, zapatos, agujeros del
   cinturón). Por defecto sólo se dibuja su contorno; con rellenar_negros se
   rellenan con contornos concéntricos separados menos de un ancho de plumón.
   SOMBREADO: zonas oscuras que no son negras (el interior de la boca) se
   rayan en zigzag con rayas más separadas que el plumón, para dar tono.
3. BORDES DE COLOR: fronteras entre regiones de color que NO tienen ya una
   línea encima (manchas de la esponja, lengua contra boca). Se cuantiza la
   imagen con k-means y se dibuja la frontera entre etiquetas lejos de la tinta.

Después se ordenan (vecino más cercano), se fusionan sólo cuando el conector
cae sobre tinta (retroceso), y cada camino pasa a serie de Fourier con un
número de waypoints proporcional a su largo real.

Todas las distancias de entrada están en mm REALES sobre el papel.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import cv2
import numpy as np
from skimage.morphology import skeletonize

from ur5_algoritmos.trayectorias import letter_trajectory_functions as L
from ur5_algoritmos.trayectorias.fourier_trajectory_functions import (
    ajustar_fourier, n_para_trazo, refinar_caminos,
)

IMAGE_DEFAULTS = {
    # Tamaño del dibujo en el papel: caja límite, se conserva la proporción.
    "ancho_mm": 160.0,
    "alto_mm": 160.0,
    "px_por_mm": 14.0,             # resolución de trabajo (14: texto de ~4 mm legible)
    "ancho_plumon_mm": 0.8,
    # 1. Líneas
    "blackhat_kernel_mm": 2.0,     # más fino que esto cuenta como línea (las líneas miden ~1.2 mm)
    "blackhat_umbral": 35,         # contraste mínimo [niveles de gris]
    "negro_v_max": 70,             # V (HSV) máximo para considerar negro
    "linea_min_mm": 0.8,           # caminos de esqueleto más cortos se descartan (tildes ~1 mm)
    # 2. Rellenos negros
    "relleno_grosor_mm": 2.0,      # negro más grueso que esto es una "zona negra"
    # False -> sólo bordes: de cada zona negra se dibuja únicamente su contorno
    # (a medio plumón del borde, por dentro). True -> se rellena con contornos
    # concéntricos y línea central.
    "rellenar_negros": False,
    "relleno_solape": 0.85,        # separación entre contornos / ancho plumón
    # 2b. Sombreado: zonas oscuras pero no negras (interior de la boca) se
    # rayan en zigzag, más separado que el plumón, para dar un tono oscuro.
    "sombreado": False,
    "sombra_gris_max": 100,        # gris máximo para sombrear
    "sombra_min_mm": 3.0,          # ancho mínimo de la zona
    "sombra_paso_mm": 1.4,         # separación entre rayas
    "sombra_angulo": 45.0,
    # 3. Bordes de color
    "bordes_color": True,
    "kmeans_k": 10,
    "borde_min_mm": 3.0,
    "borde_margen_mm": 0.3,        # no dibujar fronteras a menos de esto de la tinta
    # Orden, fusión y Fourier
    "fusion_max_mm": 1.0,          # conector recto sólo si el hueco es < esto
    "armonicos_por_cm": 10.0,
    "n_min": 4,
    "n_max": 250,
    "muestras_fourier": 1024,
    "paso_waypoint_mm": 0.6,
}


def _config(**overrides) -> SimpleNamespace:
    unknown = set(overrides) - set(IMAGE_DEFAULTS)
    if unknown:
        raise TypeError(f"Parámetros desconocidos: {sorted(unknown)}")
    c = SimpleNamespace(**{**IMAGE_DEFAULTS, **overrides})
    c.mm = lambda v: float(v) * c.px_por_mm          # mm reales -> px de trabajo
    c.pen_px = c.mm(c.ancho_plumon_mm)
    return c


def cargar_imagen(ruta: str, c) -> np.ndarray:
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    h, w = img.shape[:2]
    escala_mm = min(c.ancho_mm / w, c.alto_mm / h)          # mm por px original
    f = escala_mm * c.px_por_mm
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, (int(round(w * f)), int(round(h * f))), interpolation=interp)
    # Borde blanco: los contornos que tocan el marco quedan cerrados.
    pad = int(round(c.mm(2.0)))
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))


# -----------------------------------------------------------------------------
# 1 + 2: tinta (líneas y rellenos negros)
# -----------------------------------------------------------------------------

def separar_tinta(img: np.ndarray, c):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

    k = max(3, int(round(c.mm(c.blackhat_kernel_mm))) | 1)
    blackhat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    lineas = blackhat > c.blackhat_umbral
    negro = v < c.negro_v_max
    tinta = (lineas | negro).astype(np.uint8)
    tinta = cv2.morphologyEx(tinta, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Rellenos: zonas negras cuyo radio interior supera relleno_grosor/2.
    dt_negro = cv2.distanceTransform(negro.astype(np.uint8), cv2.DIST_L2, 5)
    r_rel = c.mm(c.relleno_grosor_mm) / 2.0
    nucleo = (dt_negro > r_rel).astype(np.uint8)
    disco = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r_rel) + 3,) * 2)
    relleno = cv2.dilate(nucleo, disco) & negro.astype(np.uint8)
    # Quitar del relleno lo que no está unido a un núcleo (bordes de líneas).
    n, lab = cv2.connectedComponents(relleno)
    if n > 1:
        con_nucleo = np.unique(lab[nucleo > 0])
        relleno = np.isin(lab, con_nucleo[con_nucleo > 0]).astype(np.uint8)

    # Las líneas se esqueletizan fuera de los rellenos (que van por contornos).
    # Se resta el relleno algo dilatado: si no, su borde (y el antialiasing)
    # queda como un anillo fino que se esqueletiza como contorno extra.
    solo_lineas = tinta & (1 - cv2.dilate(relleno, np.ones((5, 5), np.uint8)))
    return {"gris": gris, "tinta": tinta, "lineas": solo_lineas, "relleno": relleno}


def caminos_lineas(lineas: np.ndarray, c) -> list:
    esqueleto = skeletonize(lineas > 0)
    paths, graph = L.skeleton_to_paths(esqueleto, max(3, int(c.mm(c.linea_min_mm))))
    # Mapa de tinta suave para ubicar bordes con precisión sub-píxel.
    tinta = cv2.GaussianBlur(lineas.astype(np.float32), (0, 0), 0.8)
    paths = refinar_caminos(paths, tinta, graph["endpoints"], radio_plumon=c.pen_px / 2.0)
    for p in paths:
        p["tipo"] = "linea"
    return paths


def caminos_relleno(relleno: np.ndarray, c) -> list:
    """Contornos concéntricos: el primero a medio plumón del borde."""
    paths = []
    paso = c.relleno_solape * c.pen_px
    n, lab = cv2.connectedComponents(relleno)
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        comp = cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        nivel, ultimo = c.pen_px / 2.0, None
        while nivel <= dt.max() - paso / 2.0:
            cnts, _ = cv2.findContours((dt >= nivel).astype(np.uint8), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)
            for cnt in cnts:
                xy = cnt[:, 0, :].astype(float) - 1.0
                if len(xy) >= 6:
                    paths.append(L._make_path(xy[:, ::-1], closed=True, source="relleno"))
                    paths[-1]["tipo"] = "relleno"
            ultimo = nivel
            if not c.rellenar_negros:
                break                       # sólo el contorno exterior (y huecos)
            nivel += paso
        # Núcleo que el último contorno no alcanza a cubrir (zonas alargadas
        # como los agujeros del cinturón): se completa con su línea central.
        if ultimo is not None and c.rellenar_negros:
            nucleo = dt > ultimo + c.pen_px / 4.0
            if nucleo.any():
                centro, _ = L.skeleton_to_paths(skeletonize(nucleo), 2)
                for p in centro:
                    p["pixels"] = np.asarray(p["pixels"], float) - 1.0
                    p["tipo"] = "relleno"
                    paths.append(p)
        if dt.max() <= c.pen_px / 2.0:      # más fino que el plumón: un punto
            y, x = np.unravel_index(np.argmax(dt), dt.shape)
            paths.append(L._make_path(np.array([[y - 1, x - 1], [y - 1, x - 0.5]], float),
                                      source="relleno"))
            paths[-1]["tipo"] = "relleno"
    return paths


def mascara_sombra(img: np.ndarray, tinta: np.ndarray, relleno: np.ndarray, c) -> np.ndarray:
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    zona = ((gris < c.sombra_gris_max) & (v >= c.negro_v_max)).astype(np.uint8)
    zona = cv2.morphologyEx(zona, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Sólo zonas anchas: el antialiasing alrededor de las líneas queda fuera.
    dt = cv2.distanceTransform(zona, cv2.DIST_L2, 5)
    n, lab = cv2.connectedComponents(zona)
    anchas = [i for i in range(1, n) if dt[lab == i].max() >= c.mm(c.sombra_min_mm) / 2.0]
    zona = np.isin(lab, anchas).astype(np.uint8)
    # Las rayas terminan a medio plumón del borde (no invaden la línea negra).
    return cv2.erode(zona, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(c.pen_px) | 1,) * 2))


def caminos_sombreado(zona: np.ndarray, c) -> list:
    """Rayas paralelas recorridas en zigzag; se unen mientras el tramo de
    unión quede dentro de la zona, así cada zona sale en pocos trazos."""
    paths = []
    ang = np.deg2rad(c.sombra_angulo)
    d = np.array([np.cos(ang), np.sin(ang)])          # dirección de las rayas
    nrm = np.array([-d[1], d[0]])
    paso = c.mm(c.sombra_paso_mm)
    h, w = zona.shape
    n, lab = cv2.connectedComponents(zona)
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        pts = np.column_stack((xs, ys)).astype(float)
        proy_n, proy_d = pts @ nrm, pts @ d
        segmentos = []
        for k, off in enumerate(np.arange(proy_n.min() + paso / 2, proy_n.max(), paso)):
            t = np.arange(proy_d.min(), proy_d.max() + 1.0, 0.5)
            linea = off * nrm + t[:, None] * d
            xi, yi = np.rint(linea).astype(int).T
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            dentro = np.zeros(len(t), bool)
            dentro[ok] = lab[yi[ok], xi[ok]] == i
            # tramos contiguos dentro de la zona
            bordes = np.flatnonzero(np.diff(np.r_[0, dentro.astype(int), 0]))
            for a, b in zip(bordes[::2], bordes[1::2]):
                if b - a >= 3:
                    seg = linea[[a, b - 1]]
                    segmentos.append(seg if k % 2 == 0 else seg[::-1])
        actual = None
        for seg in segmentos:
            if actual is not None:
                union = np.linspace(actual[-1], seg[0], 12)
                xi, yi = np.rint(union).astype(int).T
                if np.all(lab[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)] == i):
                    actual = np.vstack([actual, seg])
                    continue
                paths.append(actual)
            actual = seg
        if actual is not None:
            paths.append(actual)
    out = []
    for xy in paths:
        p = L._make_path(xy[:, ::-1], source="sombreado")
        p["tipo"] = "sombreado"
        out.append(p)
    return out


# -----------------------------------------------------------------------------
# 3: bordes de color sin línea encima
# -----------------------------------------------------------------------------

def caminos_bordes_color(img: np.ndarray, tinta: np.ndarray, c) -> list:
    lab_img = cv2.cvtColor(cv2.medianBlur(img, 5), cv2.COLOR_BGR2LAB)
    z = lab_img.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    cv2.setRNGSeed(0)
    _, etiquetas, _ = cv2.kmeans(z, c.kmeans_k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    etiquetas = etiquetas.reshape(lab_img.shape[:2])
    # Regiones diminutas (antialiasing) se absorben con un filtro de moda.
    etiquetas = cv2.medianBlur(etiquetas.astype(np.uint8), 5)

    borde = np.zeros_like(tinta)
    borde[:, 1:] |= (etiquetas[:, 1:] != etiquetas[:, :-1]).astype(np.uint8)
    borde[1:, :] |= (etiquetas[1:, :] != etiquetas[:-1, :]).astype(np.uint8)

    dist_tinta = cv2.distanceTransform((1 - tinta).astype(np.uint8), cv2.DIST_L2, 5)
    borde &= (dist_tinta > c.mm(c.borde_margen_mm) + c.pen_px / 2.0).astype(np.uint8)

    paths, _ = L.skeleton_to_paths(skeletonize(borde > 0), max(3, int(c.mm(c.borde_min_mm))))
    # Los bordes se cortaron a cierta distancia de la tinta: los extremos que
    # quedaron cerca de ella se alargan, en su dirección, hasta tocar el
    # contorno negro (que el plumón recorre a medio ancho por dentro).
    alcance = c.mm(c.borde_margen_mm) + 1.5 * c.pen_px + 3.0
    for p in paths:
        p["tipo"] = "borde"
        if not p["closed"]:
            xy = np.asarray(p["pixels"], float)[:, ::-1]
            for al_inicio in (False, True):
                xy = _alargar_hasta_tinta(xy, tinta, alcance, c.pen_px / 2.0, al_inicio)
            p["pixels"] = xy[:, ::-1]
    return paths


def _alargar_hasta_tinta(xy, tinta, alcance, adentro, al_inicio):
    """Prolonga un extremo hasta entrar `adentro` px en la tinta, si la toca antes de `alcance`."""
    if len(xy) < 4:
        return xy
    pts = xy[::-1] if al_inicio else xy
    k = min(8, len(pts) - 1)
    d = pts[-1] - pts[-1 - k]
    n = np.linalg.norm(d)
    if n < 1e-9:
        return xy
    d /= n
    h, w = tinta.shape
    for t in np.arange(0.5, alcance, 0.5):
        x, y = np.rint(pts[-1] + t * d).astype(int)
        if not (0 <= x < w and 0 <= y < h):
            return xy
        if tinta[y, x]:
            fin = pts[-1] + (t + adentro) * d
            extra = np.linspace(pts[-1], fin, max(2, int(t + adentro) + 1))[1:]
            out = np.vstack([pts, extra])
            return out[::-1] if al_inicio else out
    return xy


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def _pixels_a_mm(px_yx: np.ndarray, c, shape) -> np.ndarray:
    """[y, x] px de trabajo -> [x, y] mm con origen en el centro de la imagen, Y hacia arriba."""
    h, w = shape
    x = (px_yx[:, 1] - (w - 1) / 2.0) / c.px_por_mm
    y = -(px_yx[:, 0] - (h - 1) / 2.0) / c.px_por_mm
    return np.column_stack((x, y))


def run_fourier_image_pipeline(ruta_imagen: str, **overrides):
    """
    Retorna (trazos, data):
      trazos: lista de dicts con 'points_mm' (Nx2, mm reales centrados en la
              imagen, Y hacia arriba), 'tipo', 'n_armonicos', 'closed'.
      data:   imágenes intermedias, tiempos y estadísticas.
    """
    c = _config(**overrides)
    t = {}
    t0 = time.perf_counter()
    img = cargar_imagen(ruta_imagen, c)
    capas = separar_tinta(img, c)
    t["tinta"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lineas = caminos_lineas(capas["lineas"], c)
    t["lineas"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rellenos = caminos_relleno(capas["relleno"], c)
    t["rellenos"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sombras = (caminos_sombreado(mascara_sombra(img, capas["tinta"], capas["relleno"], c), c)
               if c.sombreado else [])
    t["sombreado"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    bordes = caminos_bordes_color(img, capas["tinta"], c) if c.bordes_color else []
    t["bordes"] = time.perf_counter() - t0

    # Orden por vecino más cercano y fusión sólo con conectores sobre tinta.
    t0 = time.perf_counter()
    todos = lineas + rellenos + sombras + bordes
    ordenados = L.order_paths_nearest_neighbor(todos)
    gap = max(2, int(round(c.mm(c.fusion_max_mm))))
    cfg_fusion = L.make_config(
        merge_max_gap_pixels=gap, merge_max_vertical_gap_pixels=gap,
        merge_connector_points=max(1, gap // 2),
        retrace_attach_max_distance=gap, global_retrace_attach_max_distance=gap,
        retrace_max_backtrack_pixels=int(c.mm(40)), global_retrace_max_route_pixels=int(c.mm(60)),
    )
    finales = L.merge_close_ordered_paths(ordenados, cfg_fusion)
    t["orden_fusion"] = time.perf_counter() - t0

    # Fourier con waypoints proporcionales al largo real.
    t0 = time.perf_counter()
    cfg_f = SimpleNamespace(n_armonicos=0, armonicos_por_cm=c.armonicos_por_cm,
                            n_min=c.n_min, n_max=c.n_max, muestras_fourier=c.muestras_fourier,
                            px_per_mm=c.px_por_mm)
    trazos = []
    for p in finales:
        xy = L._xy(p)
        largo_px = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
        n_pts = max(4, int(np.ceil(largo_px / c.mm(c.paso_waypoint_mm))) + 1)
        curva, n = ajustar_fourier(xy, cfg_f, n_pts, p["closed"])
        if p["closed"] and len(curva) > 1:
            curva = np.vstack([curva, curva[:1]])          # cerrar el lazo
        trazos.append({
            "points_mm": _pixels_a_mm(curva[:, ::-1], c, img.shape[:2]),
            "tipo": p.get("tipo", "mixto"), "closed": p["closed"], "n_armonicos": n,
            "largo_mm": largo_px / c.px_por_mm,
        })
    t["fourier"] = time.perf_counter() - t0

    data = {
        "img": img, **capas, "config": c, "tiempos": t,
        "n_lineas": len(lineas), "n_rellenos": len(rellenos), "n_bordes": len(bordes),
        "n_sombreado": len(sombras),
        "n_antes_fusion": len(todos),
    }
    return trazos, data


__all__ = ["run_fourier_image_pipeline", "IMAGE_DEFAULTS"]
