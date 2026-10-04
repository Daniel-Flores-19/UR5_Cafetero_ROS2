"""
Igual que dibujo_completo.launch.py, pero calibrando la MESA COMPLETA.

Diferencia: en vez de calibration_draw (un solo punto, un solo Z para toda la
hoja) lanza calibration_surface_A4_4x4, que palpa 16 puntos (malla 4x4) sobre una A4 y deja
la superficie en ~/pintor_ws/src/calibracion/surface_calibration_A4.npz. move_draw_sub la carga solo y
evalúa z(x, y) en cada waypoint.

El dist_z de ~/pintor_ws/src/calibracion/z_calibrado.txt se sigue pasando como respaldo: si el .npz no
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
    ruta_z = os.path.expanduser('~/pintor_ws/src/calibracion/z_calibrado.txt')
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
        executable='img_trajectory',
        name='img_trajectory',
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
        default_value='~/pintor_ws/src/calibracion/surface_calibration_A4.npz',
        description='Mapa de la mesa que deja la calibración de superficie.')

    # Por defecto la calibración arranca desde donde esté el robot y centra la
    # malla en esa posición. Con usar_posicion_actual:=false va a q_center y
    # usa center_x/center_y (A4 vertical sobre el centro de img_trajectory.py).
    arg_actual = DeclareLaunchArgument('usar_posicion_actual', default_value='true')
    arg_cx = DeclareLaunchArgument('center_x', default_value='-0.759')
    arg_cy = DeclareLaunchArgument('center_y', default_value='-0.085')

    nodo_calibracion = Node(
        package='ur5_algoritmos',
        executable='calibration_surface_A4_4x4',
        name='calibration_surface_A4_4x4',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'usar_posicion_actual': LaunchConfiguration('usar_posicion_actual'),
            'center_x': LaunchConfiguration('center_x'),
            'center_y': LaunchConfiguration('center_y'),
            'output_prefix': '~/pintor_ws/src/calibracion/surface_calibration_A4',
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
        arg_actual,
        arg_cx,
        arg_cy,
        nodo_calibracion,
        orquestador,
    ])
