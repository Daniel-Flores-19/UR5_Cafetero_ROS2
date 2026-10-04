"""
Generación de trayectorias de dibujo.

    img_trajectory_functions        imagen -> contornos -> trayectorias
    letter_trajectory_functions     texto -> esqueleto -> splines (pipeline V7 A4)
    painting_trajectory_functions   máscara -> polígonos + relleno scanline

Se importan por módulo, porque cada uno es un pipeline independiente:

    from ur5_algoritmos.trayectorias.letter_trajectory_functions import run_full_pipeline
"""
