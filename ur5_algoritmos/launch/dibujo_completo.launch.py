import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, LogInfo, OpaqueFunction
from launch.event_handlers import OnProcessExit

def lanzar_nodo_dibujo(context, *args, **kwargs):
    # Definimos la ruta donde el nodo de calibración guardó el dato
    ruta_archivo = os.path.expanduser('~/z_calibrado.txt')
    
    # Valor por defecto por si ocurre algún fallo de lectura
    z_final = 0.1930
    
    if os.path.exists(ruta_archivo):
        try:
            with open(ruta_archivo, 'r') as f:
                z_final = float(f.read().strip())
        except Exception:
            pass

    # 1. Nodo principal de movimiento (Suscrito a los puntos)
    nodo_dibujo = Node(
        package='ur5_algoritmos',
        executable='move_draw_sub',
        name='move_draw_sub',
        output='screen',
        emulate_tty=True,
        parameters=[{'dist_z': z_final}] # <- Se pasa la información calculada
    )

    # 2. Nuevo Nodo: Generador de puntos del Nombre

    # 3. Nuevo Nodo: Generador de puntos de la Imagen
    nodo_imagen_trajectory = Node(
        package='ur5_algoritmos',  # Asegúrate de que este sea el paquete correcto
        executable='imagen_trajectory_new',
        name='imagen_trajectory_new',
        output='screen',
        emulate_tty=True
    )
    



    # Retornamos los tres nodos para que se ejecuten en paralelo tras la calibración
    return [nodo_dibujo, nodo_imagen_trajectory]


def generate_launch_description():

    # 1. DEFINICIÓN DEL NODO DE CALIBRACIÓN
    nodo_calibracion = Node(
        package='ur5_algoritmos',
        executable='calibration_draw',
        name='calibration_draw',
        output='screen',
        emulate_tty=True
    )

    # 2. MANEJADOR DE EVENTOS
    # Ejecuta el OpaqueFunction (los 3 nodos secuenciales/paralelos) al terminar la calibración
    orquestador_dibujo = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=nodo_calibracion,
            on_exit=[
                LogInfo(msg='=================================================='),
                LogInfo(msg='¡Calibración exitosa! Leyendo Z y cargando nodos de trayectoria...'),
                LogInfo(msg='=================================================='),
                OpaqueFunction(function=lanzar_nodo_dibujo) 
            ],
        )
    )

    return LaunchDescription([
        nodo_calibracion,
        orquestador_dibujo,

    ])
