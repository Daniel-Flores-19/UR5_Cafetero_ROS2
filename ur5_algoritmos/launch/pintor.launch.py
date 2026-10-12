"""
Launch único del pintor: simulación en Gazebo o robot real.

══════════════════════════ EJEMPLOS DE USO ══════════════════════════
(antes: source ~/pintor_ws/install/setup.bash)

# 1) Simulación, una palabra con Fourier (UR5e en Gazebo):
#    ros2 launch ur5_algoritmos pintor.launch.py modo:=sim modelo_robot:=ur5e \
#        trayectoria:=fourier texto:=Sergio letra_offset_y:=0.0
#
# 2) Simulación, un dibujo (sólo bordes) a 20 mm/s:
#    ros2 launch ur5_algoritmos pintor.launch.py modo:=sim modelo_robot:=ur5e \
#        trayectoria:=fourier_img imagen:=/home/mito/Downloads/bob.jpeg \
#        ancho_mm:=160 alto_mm:=160 ancho_plumon_mm:=0.8 velocidad_dibujo:=0.02
#
# 3) Robot real, PASADA EN EL AIRE (el plumón va 10 mm sobre el papel, lento)
#    para revisar el recorrido sin tocar la hoja; umbral_trazo:=0.012 hace que
#    RViz igual muestre el trazo:
#    ros2 launch ur5_algoritmos pintor.launch.py modo:=real modelo_robot:=ur5e \
#        robot_ip:=192.168.56.102 trayectoria:=fourier_img imagen:=/ruta/bob.jpeg \
#        velocidad_dibujo:=0.01 offset_tapa:=0.010 umbral_trazo:=0.012
#
# 4) Robot real, dibujo de verdad (primero 0.01 m/s, luego subir a 0.02):
#    ros2 launch ur5_algoritmos pintor.launch.py modo:=real modelo_robot:=ur5e \
#        robot_ip:=192.168.56.102 trayectoria:=fourier_img imagen:=/ruta/bob.jpeg \
#        ancho_plumon_mm:=0.8 velocidad_dibujo:=0.01 offset_contacto:=0.0004 \
#        z_min_seguridad:=0.17
#    z_min_seguridad: Z mínima de la bajada de calibración si el sensor no
#    detecta el contacto. Ponerla 1-2 cm bajo la Z del papel que muestra la
#    calibración (en el .txt de src/calibracion) para ESTE modelo_robot.
#
# Ver todos los argumentos con su descripción:
#    ros2 launch ur5_algoritmos pintor.launch.py --show-args
═════════════════════════════════════════════════════════════════════

modo:=sim
  Gazebo + RViz (GPU NVIDIA) con los controladores de ur_simulation_gazebo
  (pintor_sim.launch.py). NO calibra: move_draw_sub dibuja a una Z plana,
  z_sim, que por defecto es la Z del centro de la última calibración 4x4
  guardada. Gazebo no tiene sensor de fuerza, así que palpar no tendría sentido.

modo:=real
  ur_robot_driver + RViz, calibración de superficie 4x4 y, si termina bien,
  dibujo siguiendo z(x, y) del .npz recién escrito (como pintor_completo).
  Con lanzar_driver:=false se usa un driver que ya esté corriendo.

modelo_robot:=ur5 | ur5e
  Un solo parámetro para todo: el robot que carga Gazebo / el driver, y los
  DH que usan TODOS los nodos para la FK y el Jacobiano (variable de entorno
  PINTOR_UR_MODELO). Si no coincide con el robot de verdad, el dibujo sale
  desplazado y deformado. La calibración 4x4 guardada depende del modelo con
  que se midió: al cambiar de modelo, recalibrar (en modo real siempre se hace).

trayectoria: painting | img | letter | fourier | fourier_img (nodo que publica los XY a dibujar).
"""
import os
import time

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

CALIB_DIR = os.path.expanduser('~/pintor_ws/src/calibracion')
SUPERFICIE_PREFIX = os.path.join(CALIB_DIR, 'surface_calibration_A4')
SUPERFICIE_NPZ = SUPERFICIE_PREFIX + '.npz'
Z_POR_DEFECTO = 0.1918
JUNTAS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

TRAYECTORIAS = {
    'painting': 'painting_trajectory',
    'img': 'img_trajectory',
    'letter': 'letter_trajectory',
    'fourier': 'fourier_trajectory',   # letter con series de Fourier en vez de splines
    'fourier_img': 'fourier_image_trajectory',   # imagen (line art) con series de Fourier
}


def arg(context, nombre):
    return LaunchConfiguration(nombre).perform(context).strip()


def z_guardada():
    """Z del centro de la última calibración: .npz, luego el .txt, luego fija."""
    try:
        return float(np.load(SUPERFICIE_NPZ)['z_centro']), SUPERFICIE_NPZ
    except Exception:
        pass
    ruta_txt = os.path.join(CALIB_DIR, 'z_calibrado_superficie.txt')
    try:
        with open(ruta_txt) as f:
            return float(f.read().strip()), ruta_txt
    except Exception:
        return Z_POR_DEFECTO, 'valor por defecto'


def params_move_draw(context):
    """Parámetros de move_draw_sub que se ajustan desde el launch."""
    return {
        'modo_seguimiento': arg(context, 'modo_seguimiento'),
        'velocidad_dibujo': float(arg(context, 'velocidad_dibujo')),
        'error_max_avance': float(arg(context, 'error_max_avance')),
        'offset_contacto': float(arg(context, 'offset_contacto')),
        'offset_tapa': float(arg(context, 'offset_tapa')),
        'altura_regreso': float(arg(context, 'altura_regreso')),
    }


def nodo_trazo(context, z, umbral_defecto, sim):
    umbral = arg(context, 'umbral_trazo')
    return Node(
        package='ur_simulation_gazebo',
        executable='trazo_rviz.py',
        name='trazo_rviz',
        output='screen',
        parameters=[{'use_sim_time': sim, 'z_papel': z,
                     'umbral': float(umbral) if umbral else umbral_defecto}],
    )


def nodos_dibujo(context, sim, parametros_dibujo):
    trayectoria = arg(context, 'trayectoria')
    if trayectoria not in TRAYECTORIAS:
        raise RuntimeError(
            f"trayectoria:={trayectoria} no válida; usa {' | '.join(TRAYECTORIAS)}")
    comunes = {'use_sim_time': sim}

    # Argumentos vacíos = los valores por defecto del nodo.
    params_trayectoria = dict(comunes)
    if trayectoria == 'fourier':
        params_trayectoria['n_armonicos'] = int(arg(context, 'n_armonicos'))
    if trayectoria == 'fourier_img':
        params_trayectoria.update(
            imagen=arg(context, 'imagen'),
            ancho_mm=float(arg(context, 'ancho_mm')),
            alto_mm=float(arg(context, 'alto_mm')),
        )
    if trayectoria in ('fourier', 'fourier_img'):
        params_trayectoria['ancho_plumon_mm'] = float(arg(context, 'ancho_plumon_mm'))
    if trayectoria in ('letter', 'fourier'):
        texto = LaunchConfiguration('texto').perform(context)
        offset_y = arg(context, 'letra_offset_y')
        if texto:
            params_trayectoria['text'] = texto
        if offset_y:
            params_trayectoria['offset_y'] = float(offset_y)

    return [
        Node(
            package='ur5_algoritmos',
            executable='move_draw_sub',
            name='move_draw_sub',
            output='screen',
            emulate_tty=True,
            # Reloj de pared también en sim: con use_sim_time su temporizador de
            # 250 Hz sólo se dispara cuando Gazebo publica /clock (~15 veces por
            # segundo), y el dibujo iba a ~1 mm/s.
            parameters=[{**comunes, **parametros_dibujo, **params_move_draw(context),
                         'use_sim_time': False}],
            # Matrices de 6x6: los hilos de OpenBLAS sólo agregan sobrecarga
            # (llegaba a usar >5 núcleos sin ir más rápido).
            additional_env={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                            'MKL_NUM_THREADS': '1'},
        ),
        Node(
            package='ur5_algoritmos',
            executable=TRAYECTORIAS[trayectoria],
            name=TRAYECTORIAS[trayectoria],
            output='screen',
            emulate_tty=True,
            parameters=[params_trayectoria],
        ),
    ]


# ───────────────────────────── modo sim ─────────────────────────────
def setup_sim(context, modelo):
    z_txt = arg(context, 'z_sim')
    if z_txt:
        z, origen = float(z_txt), 'argumento z_sim'
    else:
        z, origen = z_guardada()

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('ur_simulation_gazebo'), 'launch', 'pintor_sim.launch.py'])),
        launch_arguments={
            'ur_type': modelo,
            'z_papel': str(z),
            'largo_plumon': LaunchConfiguration('largo_plumon'),
            'gazebo_gui': LaunchConfiguration('gazebo_gui'),
            'usar_gpu': LaunchConfiguration('usar_gpu'),
            # El trazo lo lanza este archivo (con umbral_trazo), no pintor_sim.
            'trazo': 'false',
        }.items(),
    )

    dibujo = nodos_dibujo(context, sim=True, parametros_dibujo={
        'dist_z': z,
        'usar_superficie': False,
    })

    # El brazo cede por gravedad entre el spawn y la activación de los
    # controladores (9 a 30 mm según la carga de la PC). Por eso, antes de
    # dibujar se lo devuelve a la postura inicial con el controlador ya activo:
    # así arranca siempre a la misma altura sobre el papel.
    ruta_q0 = os.path.join(get_package_share_directory('ur_simulation_gazebo'),
                           'config', f'initial_positions_pintor_{modelo}.yaml')
    with open(ruta_q0) as f:
        q0 = yaml.safe_load(f)
    goal = ('{trajectory: {joint_names: [' + ', '.join(JUNTAS) + '], points: [{positions: ['
            + ', '.join(str(float(q0[j])) for j in JUNTAS)
            + '], time_from_start: {sec: 2}}]}}')
    a_postura = ExecuteProcess(
        cmd=['ros2', 'action', 'send_goal',
             '/scaled_joint_trajectory_controller/follow_joint_trajectory',
             'control_msgs/action/FollowJointTrajectory', goal],
        output='log',
    )

    return [
        LogInfo(msg=f'[pintor] modo SIM ({modelo}): sin calibración, '
                    f'Z plana = {z:.5f} m ({origen})'),
        gazebo,
        nodo_trazo(context, z, 0.0015, sim=True),
        # Dar tiempo a Gazebo a spawnear el robot y activar los controladores.
        TimerAction(period=float(arg(context, 'espera_sim')), actions=[
            LogInfo(msg='[pintor] Llevando el robot a la postura inicial...'),
            a_postura,
        ]),
        RegisterEventHandler(OnProcessExit(
            target_action=a_postura,
            on_exit=[LogInfo(msg='[pintor] Postura inicial lista. Lanzando el dibujo.'), *dibujo],
        )),
    ]


# ───────────────────────────── modo real ────────────────────────────
def lanzar_dibujo_real(context, t_inicio):
    # La calibración sale con código 0 también cuando falla, así que lo que
    # distingue una corrida buena es un .npz escrito DESPUÉS de arrancarla.
    if not os.path.exists(SUPERFICIE_NPZ):
        return [LogInfo(msg=f'ERROR: no existe {SUPERFICIE_NPZ}: la calibración no terminó. '
                            'No se lanza el dibujo.')]
    mtime = os.path.getmtime(SUPERFICIE_NPZ)
    if mtime < t_inicio:
        return [LogInfo(msg=(
            f'ERROR: {SUPERFICIE_NPZ} es de una calibración anterior '
            f'({time.strftime("%H:%M:%S", time.localtime(mtime))}): la de ahora falló '
            'o se interrumpió. No se lanza el dibujo.'))]

    z, _ = z_guardada()   # respaldo plano si move_draw_sub no pudiera leer el .npz
    return [
        LogInfo(msg='¡Calibración exitosa! Cargando nodos de dibujo...'),
        # Trazo en RViz a la Z del centro de la mesa recién calibrada; el
        # desnivel medido (~1 mm) cabe en un umbral de 3 mm (el levantamiento
        # entre trazos es ~11 mm).
        nodo_trazo(context, z, 0.003, sim=False),
        *nodos_dibujo(context, sim=False, parametros_dibujo={
            'dist_z': z,
            'superficie_npz': SUPERFICIE_NPZ,
            'usar_superficie': True,
        }),
    ]


def setup_real(context, modelo):
    acciones = [LogInfo(msg=f'[pintor] modo REAL ({modelo}): driver UR + calibración 4x4 + dibujo')]

    if arg(context, 'lanzar_driver').lower() in ('true', '1'):
        acciones.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('ur_robot_driver'), 'launch', 'ur_control.launch.py'])),
            launch_arguments={
                'ur_type': modelo,
                'robot_ip': LaunchConfiguration('robot_ip'),
                # RViz propio (abajo) con el display del trazo.
                'launch_rviz': 'false',
            }.items(),
        ))

    acciones.append(Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', PathJoinSubstitution(
            [FindPackageShare('ur_simulation_gazebo'), 'config', 'pintor.rviz'])],
    ))

    calibracion = Node(
        package='ur5_algoritmos',
        executable='calibration_surface_A4_4x4',
        name='calibration_surface_A4_4x4',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'usar_posicion_actual': True,
            'output_prefix': SUPERFICIE_PREFIX,
            # Tope de seguridad: si el sensor no detecta el contacto, la bajada
            # se corta al llegar a esta Z (absoluta, en el marco de la FK).
            'z_min_seguridad': float(arg(context, 'z_min_seguridad')),
        }],
    )

    # Referencia para saber si el .npz lo escribió ESTA calibración.
    t_inicio = time.time()
    acciones += [
        TimerAction(period=float(arg(context, 'espera_driver')), actions=[calibracion]),
        RegisterEventHandler(OnProcessExit(
            target_action=calibracion,
            on_exit=[
                LogInfo(msg='Calibración terminada. Verificando la superficie...'),
                OpaqueFunction(function=lanzar_dibujo_real, kwargs={'t_inicio': t_inicio}),
            ],
        )),
    ]
    return acciones


def launch_setup(context, *args, **kwargs):
    modelo = arg(context, 'modelo_robot').lower()
    if modelo not in ('ur5', 'ur5e'):
        raise RuntimeError(f"modelo_robot:={modelo} no válido; usa ur5 | ur5e")
    # Todos los nodos de este launch (FK, Jacobiano, IK) usan los DH de este modelo.
    entorno = [SetEnvironmentVariable('PINTOR_UR_MODELO', modelo)]

    modo = arg(context, 'modo')
    if modo == 'sim':
        return entorno + setup_sim(context, modelo)
    if modo == 'real':
        return entorno + setup_real(context, modelo)
    raise RuntimeError(f"modo:={modo} no válido; usa sim | real")


def generate_launch_description():
    A = DeclareLaunchArgument
    return LaunchDescription([
        # ── General ──────────────────────────────────────────────────
        A('modo', default_value='sim', choices=['sim', 'real'],
          description='sim (Gazebo, sin calibrar) | real (driver UR + calibración 4x4).'),
        A('modelo_robot', default_value='ur5e', choices=['ur5', 'ur5e'],
          description='Robot de Gazebo/driver y DH de la FK de todos los nodos.'),
        A('trayectoria', default_value='painting',
          description='painting | img | letter | fourier | fourier_img'),

        # ── Qué dibujar ──────────────────────────────────────────────
        A('imagen', default_value='',
          description='fourier_img: ruta de la imagen.'),
        A('ancho_mm', default_value='160.0',
          description='fourier_img: ancho máximo del dibujo en el papel [mm].'),
        A('alto_mm', default_value='160.0',
          description='fourier_img: alto máximo del dibujo en el papel [mm].'),
        A('ancho_plumon_mm', default_value='0.8',
          description='fourier / fourier_img: ancho REAL de la punta [mm].'),
        A('n_armonicos', default_value='0',
          description='fourier: armónicos por trazo; 0 = automático según el largo.'),
        A('texto', default_value='',
          description='letter / fourier: palabra a escribir (vacío = la del nodo).'),
        A('letra_offset_y', default_value='',
          description='letter / fourier: offset_y [m] del texto (vacío = el del nodo, -0.15).'),

        # ── Cómo dibujar (move_draw_sub) ─────────────────────────────
        A('velocidad_dibujo', default_value='0.02',
          description='Velocidad del plumón sobre el papel [m/s]. Real: empezar con 0.01.'),
        A('modo_seguimiento', default_value='velocidad', choices=['velocidad', 'puntos'],
          description="'velocidad' (constante, nuevo) | 'puntos' (salta de waypoint en waypoint)."),
        A('error_max_avance', default_value='0.0005',
          description='Si el robot se atrasa más que esto [m], la referencia espera.'),
        A('offset_contacto', default_value='0.0004',
          description='Sube [m] la Z calibrada (la calibración mide con el plumón comprimido).'),
        A('offset_tapa', default_value='0.0',
          description='Sube [m] todo el dibujo. 0.010 = pasada en el aire, 10 mm sobre el papel.'),
        A('altura_regreso', default_value='0.03',
          description='Al terminar: sube esto [m] en vertical antes de volver al inicio.'),
        A('umbral_trazo', default_value='',
          description='Trazo en RViz: altura [m] sobre el papel que cuenta como dibujo '
                      '(vacío = 0.0015 sim / 0.003 real; para pasada en el aire usar ~0.012).'),

        # ── Sólo sim ─────────────────────────────────────────────────
        A('z_sim', default_value='',
          description='Z plana [m]; vacío = la del centro de la última calibración.'),
        A('largo_plumon', default_value='0.0',
          description='Sólo ubica la hoja en Gazebo (brida-punta) [m].'),
        A('gazebo_gui', default_value='true'),
        A('usar_gpu', default_value='true',
          description='Gazebo y RViz en la GPU NVIDIA (PRIME offload).'),
        A('espera_sim', default_value='10.0',
          description='Segundos para que Gazebo arranque antes de dibujar.'),

        # ── Sólo real ────────────────────────────────────────────────
        A('robot_ip', default_value='192.168.56.102'),
        A('lanzar_driver', default_value='true',
          description='false = usar un ur_robot_driver que ya esté corriendo.'),
        A('espera_driver', default_value='5.0',
          description='Segundos antes de lanzar la calibración (dar play a External Control).'),
        A('z_min_seguridad', default_value='0.10',
          description='Calibración: Z mínima [m] (absoluta, marco de la FK) a la que puede bajar si '
                      'el sensor no detecta el contacto. Ponerla ~1-2 cm bajo el papel; depende '
                      'de modelo_robot (ur5 y ur5e dan Z distintas para la misma postura).'),

        OpaqueFunction(function=launch_setup),
    ])
