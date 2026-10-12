#!/usr/bin/env python3
"""
fourier_trajectory.py — como letter_trajectory, pero cada trazo es una serie de
Fourier en vez de un spline (ver trayectorias/fourier_trajectory_functions.py).

Hereda todo de LetterTrajectoryPublisher (centro desde la posición del robot,
escala a la caja de dibujo, tópicos letter_trajectory/xy y /flags), así que
move_draw_sub no nota la diferencia. Sólo cambia el último paso del pipeline.

Parámetros extra: n_armonicos (0 = automático), armonicos_por_cm, n_min, n_max,
muestras_fourier, refinar_esqueleto (centrado sub-píxel + extremos) y
ancho_plumon_mm.
"""
from __future__ import annotations

import time

import rclpy
from rclpy.executors import ExternalShutdownException

from ur5_algoritmos import letter_trajectory as lt
from ur5_algoritmos.trayectorias.fourier_trajectory_functions import (
    FOURIER_DEFAULTS, run_fourier_pipeline,
)


class FourierTrajectoryPublisher(lt.LetterTrajectoryPublisher):

    def _leer_parametros_fourier(self) -> None:
        # Se declaran aquí y no en __init__: con draw_center_x/y explícitos el
        # nodo base genera la trayectoria dentro de su propio __init__.
        for name, value in FOURIER_DEFAULTS.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, value)
            setattr(self.p, name, self.get_parameter(name).value)
        # Radio del plumón (mm reales) -> píxeles de la hoja virtual, donde
        # trabaja el esqueleto: se usa al alargar los extremos de los trazos.
        if not self.has_parameter("ancho_plumon_mm"):
            self.declare_parameter("ancho_plumon_mm", 0.8)
        if self.p.radio_plumon_px <= 0.0:
            mm_real_por_mm_hoja = self._draw_scale() * 1000.0
            self.p.radio_plumon_px = (self.get_parameter("ancho_plumon_mm").value / 2.0
                                      / mm_real_por_mm_hoja * float(self.p.px_per_mm))

    def _generate_trajectories(self) -> list:
        self._leer_parametros_fourier()
        self.get_logger().info(f"[Fourier] Texto: '{self.p.text}' | Fuente: {self.p.font_path}")
        t0 = time.perf_counter()
        trazos, trazos_mm, data, graph_info = run_fourier_pipeline(
            self.p.text, self.p.font_path,
            **{key: getattr(self.p, key) for key in lt.DEFAULTS},
            **{key: getattr(self.p, key) for key in FOURIER_DEFAULTS},
        )
        dt = (time.perf_counter() - t0) * 1000.0
        self._log_pipeline_stats(trazos, data, graph_info)
        self.get_logger().info(
            f"[Fourier] {len(trazos)} trazos en {dt:.0f} ms | armónicos por trazo: "
            f"{[t['n_armonicos'] for t in trazos]}")
        return [self._mm_to_robot_xy(pts) for pts in trazos_mm if len(pts)]


def main(args=None):
    rclpy.init(args=args)
    node = FourierTrajectoryPublisher()
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
