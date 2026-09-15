"""
Igual que dibujo_completo.launch.py, pero calibrando la MESA COMPLETA.

Diferencia: en vez de calibration_draw (un solo punto, un solo Z para toda la
hoja) lanza calibration_surface_A4_3x3, que palpa 9 puntos sobre una A4 y deja
la superficie en ~/pintorV2_ws/calibracion/surface_calibration_A4.npz. move_draw_sub la carga solo y
evalúa z(x, y) en cada waypoint.

El dist_z de ~/pintorV2_ws/calibracion/z_calibrado.txt se sigue pasando como respaldo: si el .npz no
existe o no se puede leer, el dibujo cae al comportamiento de siempre.
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, LogInfo, OpaqueFunction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit


def lanzar_nodos_dibujo(context, *args, **kwargs):
    # Respaldo por si la superficie no está disponible.
    ruta_z = os.path.expanduser('~/pintorV2_ws/calibracion/z_calibrado.txt')
    z_final = 0.1930

    if os.path.exists(ruta_z):
        try:
            with open(ruta_z, 'r') as f:
                z_final = float(f.read().strip())
        except Exception:
            pass

    npz = os.path.expanduser(
        LaunchConfiguration('superficie_npz').perform(context))

    if not os.path.exists(npz):
        return [LogInfo(msg=(
            f'No se encontró {npz}: la calibración de superficie no terminó. '
            'No se lanza el dibujo.'))]

    nodo_dibujo = Node(
        package='ur5_algoritmos',
        executable='move_draw_sub',
        name='move_draw_sub',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'dist_z': z_final,
            'superficie_npz': npz,
            'usar_superficie': True,
        }],
    )

    nodo_imagen = Node(
        package='ur5_algoritmos',
        executable='imagen_trajectory',
        name='imagen_trajectory',
        output='screen',
        emulate_tty=True,
    )

    return [
        LogInfo(msg=f'Superficie cargada desde {npz}'),
        nodo_dibujo,
        nodo_imagen,
    ]


def generate_launch_description():

    arg_npz = DeclareLaunchArgument(
        'superficie_npz',
        default_value='~/pintorV2_ws/calibracion/surface_calibration_A4.npz',
        description='Mapa de la mesa que deja la calibración de superficie.')

    # Centro y tamaño de la hoja: por defecto A4 vertical sobre el centro de
    # dibujo que usa imagen_trajectory.py.
    arg_cx = DeclareLaunchArgument('center_x', default_value='-0.759')
    arg_cy = DeclareLaunchArgument('center_y', default_value='-0.085')

    nodo_calibracion = Node(
        package='ur5_algoritmos',
        executable='calibration_surface_A4_3x3',
        name='calibration_surface_A4_3x3',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'center_x': LaunchConfiguration('center_x'),
            'center_y': LaunchConfiguration('center_y'),
            'output_prefix': '~/pintorV2_ws/calibracion/surface_calibration_A4',
        }],
    )

    orquestador = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=nodo_calibracion,
            on_exit=[
                LogInfo(msg='=================================================='),
                LogInfo(msg='Calibración de superficie terminada. Cargando dibujo...'),
                LogInfo(msg='=================================================='),
                OpaqueFunction(function=lanzar_nodos_dibujo),
            ],
        )
    )

    return LaunchDescription([
        arg_npz,
        arg_cx,
        arg_cy,
        nodo_calibracion,
        orquestador,
    ])
