import os
import time
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, LogInfo, OpaqueFunction
from launch.event_handlers import OnProcessExit

# Mapa de la mesa que escribe calibration_surface_A4_4x4 (su output_prefix por
# defecto + '.npz'). Sólo se escribe si la calibración termina bien.
SUPERFICIE_NPZ = os.path.expanduser('~/pintor_ws/src/calibracion/surface_calibration_A4.npz')


def lanzar_nodo_dibujo(context, *args, t_inicio=0.0, **kwargs):

    # La calibración sale con código 0 también cuando falla (punto fuera de
    # alcance, z_min_seguridad, Ctrl+C...), así que el código de salida no
    # sirve. Lo que sí distingue una corrida buena es que haya escrito el .npz
    # DESPUÉS de arrancar este launch; si no, move_draw_sub cargaría el mapa de
    # una calibración anterior (quizás con la hoja en otro sitio).
    if not os.path.exists(SUPERFICIE_NPZ):
        return [LogInfo(msg=(
            f'ERROR: no existe {SUPERFICIE_NPZ}: la calibración no terminó. '
            'No se lanza el dibujo.'))]

    if os.path.getmtime(SUPERFICIE_NPZ) < t_inicio:
        return [LogInfo(msg=(
            f'ERROR: {SUPERFICIE_NPZ} es de una calibración anterior '
            f'({time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(SUPERFICIE_NPZ)))}): '
            'la de ahora falló o se interrumpió. No se lanza el dibujo.'))]

    # Valor por defecto por si ocurre algún fallo de lectura
    z_final = 0.1930


    nodo_dibujo = Node(
        package='ur5_algoritmos',
        executable='move_draw_sub',
        name='move_draw_sub',
        output='screen',
        emulate_tty=True
    )

    nodo_imagen_trajectory = Node(
        package='ur5_algoritmos',
        executable='painting_trajectory',
        name='painting_trajectory',
        output='screen',
        emulate_tty=True
    )



    return [
        LogInfo(msg='=================================================='),
        LogInfo(msg='¡Calibración exitosa! Cargando nodos de trayectoria...'),
        LogInfo(msg='=================================================='),
        nodo_dibujo,
        nodo_imagen_trajectory,
    ]


def generate_launch_description():

    # Referencia para saber si el .npz lo escribió ESTA calibración.
    t_inicio = time.time()

    nodo_calibracion = Node(
        package='ur5_algoritmos',
        executable='calibration_surface_A4_4x4',
        name='calibration_surface_A4_4x4',
        output='screen',
        emulate_tty=True
    )

    orquestador_dibujo = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=nodo_calibracion,
            on_exit=[
                LogInfo(msg='Calibración terminada. Verificando que haya guardado la superficie...'),
                OpaqueFunction(function=lanzar_nodo_dibujo, kwargs={'t_inicio': t_inicio}),
            ],
        )
    )

    return LaunchDescription([
        nodo_calibracion,
        orquestador_dibujo,

    ])
