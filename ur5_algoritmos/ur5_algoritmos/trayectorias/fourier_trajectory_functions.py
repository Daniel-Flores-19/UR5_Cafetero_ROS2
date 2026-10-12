#!/usr/bin/env python3
"""
fourier_trajectory_functions.py — variante Fourier del pipeline de letras.

Mismo flujo que letter_trajectory_functions.run_full_pipeline (texto -> imagen
-> esqueleto -> grafo -> caminos ordenados y fusionados), pero el último paso
cambia: en vez de ajustar un spline a cada camino, se representa con una serie
de Fourier truncada a N armónicos.

- Camino ABIERTO (la mayoría de trazos del esqueleto): serie COSENO (DCT-II).
  Equivale a recorrer el trazo ida y vuelta, lo que lo vuelve una curva
  cerrada y par: no hay salto entre el final y el inicio, así que no aparece
  el fenómeno de Gibbs en los extremos.
- Camino CERRADO (o, ciclos del esqueleto, puntos de la i): serie compleja
  z(t) = x(t) + i y(t) con armónicos -N..N (los "epiciclos").

Antes de transformar, cada camino se remuestrea a paso de arco constante: así
t es proporcional a la distancia recorrida y los armónicos se reparten a lo
largo del trazo, no según cómo cayeron los píxeles.

Uso:
    trazos, trazos_mm, data, graph_info = run_fourier_pipeline(texto, fuente,
                                                               n_armonicos=0, **overrides)
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dct, fft
from scipy.ndimage import map_coordinates, uniform_filter1d

from ur5_algoritmos.trayectorias import letter_trajectory_functions as L

FOURIER_DEFAULTS = {
    # Armónicos por trazo. 0 = automático: armonicos_por_cm * largo del trazo
    # (en cm de la hoja virtual A4), acotado a [n_min, n_max].
    "n_armonicos": 0,
    "armonicos_por_cm": 6.0,
    "n_min": 6,
    "n_max": 150,
    # Muestras a paso de arco constante antes de transformar.
    "muestras_fourier": 512,
    # Mejoras del esqueleto (ver refinar_caminos):
    "refinar_esqueleto": True,
    # Radio de la punta del plumón en píxeles de la imagen de trabajo. Los
    # extremos se alargan hasta el final de la tinta MENOS este radio, para
    # que la punta redonda no se pase del trazo original.
    "radio_plumon_px": 0.0,
}


# =============================================================================
# Mejoras del esqueleto
# =============================================================================
#
# El esqueleto de skimage es una cadena de píxeles enteros que (1) se queda a
# medio grosor de cada punta del trazo y (2) puede caer hasta medio píxel
# fuera del centro. Ambas cosas se corrigen mirando la imagen original:
#
# - centrar_camino: en cada punto se mira el perfil de tinta a lo largo de la
#   normal, se ubican los dos bordes con interpolación (sub-píxel) y el punto
#   se mueve al punto medio.
# - extender_extremo: desde cada extremo LIBRE (grado 1 en el grafo, no una
#   bifurcación) se avanza en la dirección del trazo hasta que se acaba la
#   tinta, y se alarga el camino hasta ahí menos el radio del plumón.
#
# `tinta` es un mapa float en [0, 1] (1 = tinta) del mismo tamaño que la
# imagen; el borde está donde vale 0.5. Con la imagen antialiasing del texto
# (o un blur leve de una máscara) el borde sale con precisión sub-píxel.


def mapa_tinta(gris: np.ndarray) -> np.ndarray:
    """Imagen en gris (tinta oscura sobre blanco) -> tinta en [0, 1]."""
    return 1.0 - gris.astype(float) / 255.0


def _muestrear(tinta, pts_xy):
    return map_coordinates(tinta, [pts_xy[..., 1], pts_xy[..., 0]], order=1, mode="constant", cval=0.0)


def _tangentes(xy: np.ndarray, ventana: int = 7) -> np.ndarray:
    suave = uniform_filter1d(xy, size=min(ventana, len(xy)), axis=0, mode="nearest")
    t = np.gradient(suave, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.maximum(n, 1e-9)


def centrar_camino(xy: np.ndarray, tinta: np.ndarray, radio_max: float,
                   paso: float = 0.25, desplazamiento_max: float = 2.0) -> np.ndarray:
    """Mueve cada punto al centro entre los dos bordes de tinta (sub-píxel)."""
    if len(xy) < 3:
        return xy
    t = _tangentes(xy)
    normal = np.column_stack((-t[:, 1], t[:, 0]))
    offs = np.arange(paso, radio_max + paso, paso)

    def borde(signo):
        pts = xy[:, None, :] + signo * offs[None, :, None] * normal[:, None, :]
        perfil = _muestrear(tinta, pts)                       # (n, len(offs))
        fuera = perfil < 0.5
        idx = np.argmax(fuera, axis=1)                        # primer punto fuera
        ok = fuera[np.arange(len(xy)), idx] & (idx > 0)
        a = perfil[np.arange(len(xy)), np.maximum(idx - 1, 0)]
        b = perfil[np.arange(len(xy)), idx]
        frac = np.clip((a - 0.5) / np.maximum(a - b, 1e-9), 0.0, 1.0)
        return np.where(ok, offs[np.maximum(idx - 1, 0)] + frac * paso, np.nan)

    d_mas, d_menos = borde(+1.0), borde(-1.0)
    shift = (d_mas - d_menos) / 2.0
    valido = np.isfinite(shift) & (np.abs(shift) <= desplazamiento_max)
    shift = np.where(valido, shift, 0.0)
    # Suavizado leve: el desplazamiento varía despacio a lo largo del trazo.
    shift = uniform_filter1d(shift, size=min(5, len(shift)), mode="nearest")
    return xy + shift[:, None] * normal


def extender_extremo(xy: np.ndarray, tinta: np.ndarray, radio_plumon: float,
                     alcance_max: float, paso: float = 0.25, al_inicio: bool = False) -> np.ndarray:
    """Alarga un extremo libre del camino hasta el final de la tinta."""
    if len(xy) < 4:
        return xy
    pts = xy[::-1] if al_inicio else xy
    k = min(8, len(pts) - 1)
    direc = pts[-1] - pts[-1 - k]
    norma = np.linalg.norm(direc)
    if norma < 1e-9:
        return xy
    direc /= norma
    offs = np.arange(paso, alcance_max + paso, paso)
    perfil = _muestrear(tinta, pts[-1] + offs[:, None] * direc)
    fuera = np.nonzero(perfil < 0.5)[0]
    if len(fuera) == 0:
        return xy
    i = fuera[0]
    a = perfil[i - 1] if i > 0 else 1.0
    frac = np.clip((a - 0.5) / max(a - perfil[i], 1e-9), 0.0, 1.0)
    largo = (offs[i - 1] if i > 0 else 0.0) + frac * paso - radio_plumon
    if largo <= 0.5:
        return xy
    extra = pts[-1] + np.arange(1.0, largo + 1e-9, 1.0)[:, None] * direc
    if len(extra) == 0 or np.linalg.norm(extra[-1] - pts[-1]) < largo - 1e-6:
        extra = np.vstack([extra, pts[-1] + largo * direc])
    out = np.vstack([pts, extra])
    return out[::-1] if al_inicio else out


def refinar_caminos(paths: list, tinta: np.ndarray, extremos: np.ndarray,
                    radio_plumon: float = 0.0) -> list:
    """
    Aplica centrado sub-píxel y extensión de extremos libres a caminos del
    esqueleto ({pixels [y, x]}). `extremos` son los píxeles de grado 1 del grafo.
    """
    if not paths:
        return paths
    dt = cv2.distanceTransform((tinta >= 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    libres = set(map(tuple, np.asarray(extremos, dtype=int).reshape(-1, 2)))
    out = []
    for path in paths:
        px = np.asarray(path["pixels"], dtype=float)
        if len(px) < 3 or path.get("is_mark"):
            out.append(path)
            continue
        y0, x0 = np.clip(np.rint(px).astype(int), 0, np.array(dt.shape) - 1).T
        radio = float(np.percentile(dt[y0, x0], 90)) if len(px) else 2.0
        xy = centrar_camino(px[:, ::-1], tinta, radio_max=2.0 * radio + 2.0)
        if not path["closed"]:
            alcance = 2.0 * radio + 3.0
            if tuple(np.rint(px[-1]).astype(int)) in libres:
                xy = extender_extremo(xy, tinta, radio_plumon, alcance)
            if tuple(np.rint(px[0]).astype(int)) in libres:
                xy = extender_extremo(xy, tinta, radio_plumon, alcance, al_inicio=True)
        out.append({**path, "pixels": xy[:, ::-1]})
    return out


def _remuestrear_arco(xy: np.ndarray, m: int, cerrado: bool) -> tuple[np.ndarray, float]:
    """m puntos equiespaciados en longitud de arco; devuelve también el largo."""
    xy = L._remove_consecutive_duplicates(np.asarray(xy, dtype=float))
    if cerrado and np.linalg.norm(xy[0] - xy[-1]) > 1e-9:
        xy = np.vstack([xy, xy[0]])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    largo = float(s[-1])
    # Cerrado: el último punto repite el primero, así que se excluye (periódico).
    u = np.linspace(0.0, largo, m, endpoint=not cerrado)
    return np.column_stack((np.interp(u, s, xy[:, 0]), np.interp(u, s, xy[:, 1]))), largo


def n_para_trazo(largo_px: float, c) -> int:
    if c.n_armonicos > 0:
        return int(c.n_armonicos)
    largo_cm = largo_px / float(c.px_per_mm) / 10.0
    return int(np.clip(np.ceil(c.armonicos_por_cm * largo_cm), c.n_min, c.n_max))


def coeficientes_coseno(xy: np.ndarray, n: int) -> np.ndarray:
    """Primeros n coeficientes DCT-II (sin normalizar) de x e y: array (n, 2)."""
    return dct(xy, type=2, axis=0)[:n]


def evaluar_coseno(coef: np.ndarray, m: int, t: np.ndarray) -> np.ndarray:
    """
    Serie coseno en t ∈ [0, m-1] (t continuo, en unidades de muestra):
        f(t) = X0/(2m) + (1/m) Σ_{k≥1} Xk cos(π k (2t+1) / (2m))
    """
    k = np.arange(coef.shape[0])
    base = np.cos(np.pi * np.outer(2.0 * t + 1.0, k) / (2.0 * m))   # (len(t), n)
    peso = np.full(coef.shape[0], 1.0 / m)
    peso[0] = 1.0 / (2.0 * m)
    return base @ (coef * peso[:, None])


def coeficientes_complejos(xy: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Armónicos -n..n de z = x + iy: (frecuencias, coeficientes Z_k / m)."""
    m = len(xy)
    z = fft(xy[:, 0] + 1j * xy[:, 1]) / m
    n = min(n, (m - 1) // 2)
    freqs = np.r_[0:n + 1, -n:0]
    return freqs, z[freqs]


def evaluar_complejo(freqs: np.ndarray, coef: np.ndarray, s: np.ndarray) -> np.ndarray:
    """z(s) = Σ Z_k e^{2πi k s}, s ∈ [0, 1] (una vuelta completa)."""
    z = np.exp(2j * np.pi * np.outer(s, freqs)) @ coef
    return np.column_stack((z.real, z.imag))


def ajustar_fourier(points_xy, c, num_points: int, cerrado: bool) -> tuple[np.ndarray, int]:
    """Curva Fourier de num_points puntos para un camino; devuelve (curva, N)."""
    points_xy = np.asarray(points_xy, dtype=float)
    if len(L._remove_consecutive_duplicates(points_xy)) < 3:
        return L.linear_resample(points_xy, num_points=num_points), 0

    m = int(c.muestras_fourier)
    xy, largo = _remuestrear_arco(points_xy, m, cerrado)
    n = n_para_trazo(largo, c)

    if cerrado:
        freqs, coef = coeficientes_complejos(xy, n)
        curva = evaluar_complejo(freqs, coef, np.linspace(0.0, 1.0, num_points))
    else:
        coef = coeficientes_coseno(xy, n)
        curva = evaluar_coseno(coef, m, np.linspace(0.0, m - 1.0, num_points))
    return curva, n


def paths_to_fourier(ordered_paths: list, c) -> list:
    """Equivalente a L.paths_to_splines, con series de Fourier."""
    trazos = []
    for idx, path in enumerate(ordered_paths, start=1):
        control_points = L._xy(path)
        curva, n = ajustar_fourier(control_points, c, c.spline_points, path["closed"])
        trazo = {key: path[key] for key in
                 ("closed", "is_mark", "is_dot", "is_diaeresis", "is_accent", "mark_kind",
                  "source", "merged_count", "retrace_count", "global_retrace_count")}
        trazo.update(id=idx, control_points=control_points, points=curva, n_armonicos=n)
        trazos.append(trazo)
    return trazos


def run_fourier_pipeline(text: str, font_path: str, **overrides):
    """
    Igual que L.run_full_pipeline (mismas claves de DEFAULTS) más las de
    FOURIER_DEFAULTS. Retorna (trazos, trazos_mm, data, graph_info).
    """
    fourier = {k: overrides.pop(k, v) for k, v in FOURIER_DEFAULTS.items()}
    c = L.make_config(**overrides)
    for k, v in fourier.items():
        setattr(c, k, v)

    image_path = L.render_text_image(text, font_path, c)
    data = L.preprocess_text_image(image_path, c)

    mark_paths = L.mark_components_to_paths(data["mark_components"], data["connected_labels"], c)
    skeleton_paths, graph_info = L.skeleton_to_paths(data["skeleton_no_marks_bool"], c.min_path_pixels)
    if c.refinar_esqueleto:
        gris = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        skeleton_paths = refinar_caminos(skeleton_paths, mapa_tinta(gris),
                                         graph_info["endpoints"], c.radio_plumon_px)

    if c.postpone_marks_to_end:
        ordered_paths = (L.order_paths_for_writing(skeleton_paths, c.order_strategy)
                         + L.order_paths_for_writing(mark_paths, "left_to_right"))
    else:
        ordered_paths = L.order_paths_for_writing(skeleton_paths + mark_paths, c.order_strategy)

    final_paths = L.merge_close_ordered_paths(ordered_paths, c) if c.merge_close_paths else ordered_paths

    trazos = paths_to_fourier(final_paths, c)
    trazos_mm = [t["points"] / float(c.px_per_mm) for t in trazos]
    for t, pts_mm in zip(trazos, trazos_mm):
        t["points_mm"] = pts_mm

    data.update(
        image_path=image_path,
        raw_skeleton_paths=skeleton_paths,
        mark_paths=mark_paths,
        raw_paths=skeleton_paths + mark_paths,
        ordered_paths=ordered_paths,
        final_paths=final_paths,
        spline_arrays_mm=trazos_mm,
        roi_mm=c.roi_mm,
        px_per_mm=float(c.px_per_mm),
    )
    return trazos, trazos_mm, data, graph_info


__all__ = ["run_fourier_pipeline", "FOURIER_DEFAULTS", "paths_to_fourier", "ajustar_fourier",
           "refinar_caminos", "mapa_tinta"]
