"""
UR5 del pintor en Gazebo + RViz.

Parte de ur_sim_lab201.launch.py, con tres cambios para que los nodos de
ur5_algoritmos (escritos para el robot real) funcionen sin modificarlos:

- Controladores de config/pintor_sim_controllers.yaml: el de trayectoria se
  llama scaled_joint_trajectory_controller (pero es un JointTrajectoryController
  normal, porque Gazebo no tiene speed_scaling) y forward_position_controller
  queda cargado e inactivo, igual que con ur_robot_driver.
- Postura inicial con el plumón sobre el centro de la última calibración
  (config/initial_positions_pintor_<ur_type>.yaml).
- Una mesa con una hoja A4 cuya superficie queda a la altura z_papel (marco
  base del robot), para ver dónde dibuja.
- trazo_rviz.py: publica en /pintor/trazo lo que el plumón va dibujando y
  RViz lo muestra (config/pintor.rviz).

Uso suelto:
  ros2 launch ur_simulation_gazebo pintor_sim.launch.py z_papel:=0.1918
Normalmente lo incluye ur5_algoritmos/launch/pintor.launch.py modo:=sim.
"""
import os
import tempfile

from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Altura a la que se spawnea el robot (encima de la mesa ur5_base del world).
Z_BASE_ROBOT = 0.65

# Centro de la hoja en el marco DH de fkine_ur5 (= base_link_inertia), el de la
# última calibración 4x4. base_link_inertia está girado pi en Z respecto a
# base_link, que es el marco del mundo de Gazebo (robot en el origen, sin yaw):
# por eso X e Y cambian de signo al pasar al mundo.
CENTRO_HOJA_DH = (0.6481, -0.1121)
HOJA = (0.210, 0.297)          # A4: 210 mm en X, 297 mm en Y (como la calibración)
MESA = (0.45, 0.60, 0.03)      # tablero: largo X, ancho Y, espesor


def sdf_mesa(alto_tablero):
    """Mesa con patas hasta el piso; el origen del modelo es la cara de la hoja."""
    lx, ly, esp = MESA
    hx, hy = HOJA
    # La hoja es solo visual; el tablero (con colisión) queda 1 mm por debajo
    # para que la brida no choque con él cuando el plumón "toca" el papel.
    z_tab = -0.001 - esp / 2
    largo_pata = max(alto_tablero - esp, 0.01)
    z_pata = -0.001 - esp - largo_pata / 2
    patas = ''
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        patas += f"""
      <visual name='pata{i}'>
        <pose>{sx * (lx / 2 - 0.03)} {sy * (ly / 2 - 0.03)} {z_pata} 0 0 0</pose>
        <geometry><box><size>0.04 0.04 {largo_pata}</size></box></geometry>
        <material><ambient>0.3 0.3 0.3 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material>
      </visual>"""
    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='mesa_pintor'>
    <static>true</static>
    <link name='link'>
      <collision name='tablero'>
        <pose>0 0 {z_tab} 0 0 0</pose>
        <geometry><box><size>{lx} {ly} {esp}</size></box></geometry>
      </collision>
      <visual name='tablero'>
        <pose>0 0 {z_tab} 0 0 0</pose>
        <geometry><box><size>{lx} {ly} {esp}</size></box></geometry>
        <material><ambient>0.55 0.4 0.25 1</ambient><diffuse>0.55 0.4 0.25 1</diffuse></material>
      </visual>
      <visual name='hoja_A4'>
        <pose>0 0 -0.0005 0 0 0</pose>
        <geometry><box><size>{hx} {hy} 0.001</size></box></geometry>
        <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse></material>
      </visual>{patas}
    </link>
  </model>
</sdf>
"""


def launch_setup(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type")
    launch_rviz = LaunchConfiguration("launch_rviz")
    gazebo_gui = LaunchConfiguration("gazebo_gui")

    pkg = FindPackageShare("ur_simulation_gazebo")
    controllers_file = PathJoinSubstitution([pkg, "config", "pintor_sim_controllers.yaml"])
    # Postura de arranque calculada con los DH de cada modelo.
    initial_positions = PathJoinSubstitution(
        [pkg, "config", ["initial_positions_pintor_", ur_type, ".yaml"]])
    world_file = PathJoinSubstitution([pkg, "world", "lab_base_world_classic.sdf"])
    # view_robot.rviz de ur_description + el MarkerArray /pintor/trazo.
    rviz_config_file = PathJoinSubstitution([pkg, "config", "pintor.rviz"])

    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution([FindPackageShare("ur_description"), "urdf", "ur.urdf.xacro"]), " ",
        "safety_limits:=true ",
        "safety_pos_margin:=0.15 ",
        "safety_k_position:=20 ",
        "name:=ur ",
        "ur_type:=", ur_type, " ",
        'prefix:="" ',
        "sim_gazebo:=true ",
        "simulation_controllers:=", controllers_file, " ",
        "initial_positions_file:=", initial_positions,
    ])
    robot_description = {"robot_description": robot_description_content}

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"use_sim_time": True}, robot_description],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(launch_rviz),
    )

    def spawner(nombre, *extra):
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[nombre, "--controller-manager", "/controller_manager", *extra],
        )

    joint_state_broadcaster_spawner = spawner("joint_state_broadcaster")
    trayectoria_spawner = spawner("scaled_joint_trajectory_controller")
    forward_pos_spawner = spawner("forward_position_controller", "--inactive")

    delay_rviz = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rviz_node],
        ),
        condition=IfCondition(launch_rviz),
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("gazebo_ros"), "/launch", "/gazebo.launch.py"]),
        launch_arguments={"gui": gazebo_gui, "world": world_file}.items(),
    )

    gazebo_spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_ur",
        arguments=["-entity", "ur", "-topic", "robot_description",
                   "-x", "0.0", "-y", "0.0", "-z", str(Z_BASE_ROBOT), "-timeout", "120"],
        output="screen",
    )

    # El world usa model://ur5_base, surgery_table y bidon. Si no están en
    # GAZEBO_MODEL_PATH, Gazebo los busca en internet y se cuelga sin levantar
    # /spawn_entity.
    modelos = os.path.join(pkg.perform(context), "models")

    trazo_node = Node(
        package="ur_simulation_gazebo",
        executable="trazo_rviz.py",
        name="trazo_rviz",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "z_papel": float(LaunchConfiguration("z_papel").perform(context)),
        }],
    )

    # Laptop híbrida (Intel UHD + NVIDIA GTX 1650, PRIME on-demand): sin estas
    # variables gzclient, gzserver y RViz renderizan en la Intel. Con ellas
    # OpenGL/Vulkan van a la NVIDIA. Afecta sólo a los procesos de este launch.
    gpu = []
    if LaunchConfiguration("usar_gpu").perform(context).lower() in ("true", "1"):
        gpu = [
            SetEnvironmentVariable("__NV_PRIME_RENDER_OFFLOAD", "1"),
            SetEnvironmentVariable("__GLX_VENDOR_LIBRARY_NAME", "nvidia"),
            SetEnvironmentVariable("__VK_LAYER_NV_optimus", "NVIDIA_only"),
        ]

    usar_trazo = LaunchConfiguration("trazo").perform(context).lower() in ("true", "1")
    nodos = gpu + [
        AppendEnvironmentVariable("GAZEBO_MODEL_PATH", modelos),
        *([trazo_node] if usar_trazo else []),
        robot_state_publisher_node,
        joint_state_broadcaster_spawner,
        delay_rviz,
        trayectoria_spawner,
        forward_pos_spawner,
        gazebo,
        gazebo_spawn_robot,
    ]

    if LaunchConfiguration("mesa").perform(context).lower() in ("true", "1"):
        # z_papel es la Z de la BRIDA (lo que da fkine_ur5) cuando el plumón
        # toca; la hoja está largo_plumon más abajo.
        z_papel = float(LaunchConfiguration("z_papel").perform(context))
        largo_plumon = float(LaunchConfiguration("largo_plumon").perform(context))
        z_hoja_mundo = Z_BASE_ROBOT + z_papel - largo_plumon

        ruta_sdf = os.path.join(tempfile.gettempdir(), "pintor_mesa.sdf")
        with open(ruta_sdf, "w") as f:
            f.write(sdf_mesa(z_hoja_mundo))

        nodos.append(Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            name="spawn_mesa_pintor",
            arguments=["-entity", "mesa_pintor", "-file", ruta_sdf,
                       "-x", str(-CENTRO_HOJA_DH[0]), "-y", str(-CENTRO_HOJA_DH[1]),
                       "-z", str(z_hoja_mundo), "-timeout", "120"],
            output="screen",
        ))

    return nodos


def generate_launch_description():
    return LaunchDescription([
        # ur5 | ur5e. Debe ser el mismo modelo que usan los nodos para la FK
        # (PINTOR_UR_MODELO); pintor.launch.py pasa ambos desde modelo_robot.
        DeclareLaunchArgument("ur_type", default_value="ur5", choices=["ur5", "ur5e"]),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("trazo", default_value="true",
                              description="Lanzar trazo_rviz.py (pintor.launch.py lanza el suyo)."),
        DeclareLaunchArgument("usar_gpu", default_value="true",
                              description="Renderizar Gazebo y RViz en la NVIDIA (PRIME offload)."),
        DeclareLaunchArgument("mesa", default_value="true",
                              description="Spawnear la mesa con la hoja A4."),
        DeclareLaunchArgument("z_papel", default_value="0.1918",
                              description="Z de la brida [m, marco base] con el plumón en el papel."),
        DeclareLaunchArgument("largo_plumon", default_value="0.0",
                              description="Distancia brida-punta [m]; baja la hoja esa cantidad."),
        OpaqueFunction(function=launch_setup),
    ])
