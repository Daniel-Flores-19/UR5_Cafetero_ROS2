# UR5_Cafetero_ROS2

## Project overview

This ros2 pkg vectorizes an image and creates paths for a robotic arm, such as a UR5, to draw in a 2D plane.

## System diagram

![Nodes and Topics graph](imgs/img_system.png)

## Prerequisites and dependencies

This project requires the following software:
- Ubuntu 22.04
- ROS 2 [Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- Gazebo Classic
- Docker [Engine](https://docs.docker.com/engine/install/ubuntu/) (Docker Desktop is not required on Linux; the Engine + CLI is enough to build and run the image below)

If you don't want to use docker:
- UR5 official [repo](https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation)
- Gazebo [Fortress](https://gazebosim.org/docs/latest/ros_installation/)

### Python dependencies

The trajectory-generation pipeline (`letter_trajectory_functions_img.py`, `imagen_trajectory_new.py`) additionally requires:
- `numpy`
- `opencv-python`
- `Pillow`
- `scikit-image` (imported as `skimage` — note this is a different PyPI package name than the import name; `pip install skimage` will fail)
- `matplotlib`

If installing manually (outside Docker):
```bash
pip3 install --user numpy opencv-python Pillow scikit-image matplotlib
```
If you hit a `UserWarning: Unable to import Axes3D` from matplotlib, it usually means both an apt-installed `python3-matplotlib` and a pip `--user` install are present on `sys.path`. This is generally harmless, but if you want a clean environment, install matplotlib in a virtual environment created with `--system-site-packages` instead of `pip install --user`.

## Installation instructions

After installing the required packages from the UR5 official repo, we just need to clone this repo into the same workspace
```bash
cd ros2_ws/src
git clone https://github.com/Daniel-Flores-19/UR5_Cafetero_ROS2.git
cd ..
colcon build
source install/setup.bash
```
This will allow the user to use this entire project with/without docker

## Usage 
### With Docker
To use it with docker we need to build the image inside the repo.
```bash
cd ros2_ws/src/UR5_Cafetero_ROS2
docker build -t ur5_draw .
```
Then, we should be able to see the ID and the name of the image
```bash
docker ps
```
To run a terminal inside the container, it is recommended to add local connections to enable RVIZ2 and Gazebo
```bash
xhost +local:docker
docker run -it --network=host --ipc=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw --env=DISPLAY ur5_draw:latest
```

Once inside the container, we should be able to see all the files and build the workspace with all the packages.
```bash
cd ros2_ws
colcon build
source install/setup.bash
```

Finally, you need three terminals inside the container:

**Terminal 1 — Gazebo + UR5 control:**
```bash
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py
```

**Terminal 2 — calibration + drawing pipeline:**
```bash
ros2 launch ur5_algoritmos dibujo_completo.launch.py
```

**Terminal 3 — simulated force feedback:**

Because this URDF does not include a force/torque sensor, `calibration_draw` will wait indefinitely on `/force_torque_sensor_broadcaster/wrench` unless you publish a fake contact message yourself. This topic is only advertised once `calibration_draw` (Terminal 2) has started and subscribed to it, so wait until you see the calibration node's descent logs (`"¡Iniciando calibracion!"` / repeated position output) before publishing:
```bash
ros2 topic pub /force_torque_sensor_broadcaster/wrench geometry_msgs/msg/WrenchStamped \
"{wrench: {force: {x: 0.0, y: 0.0, z: -3.5}}}" --once
```
Publishing this too early (before the end-effector begins its descent) will cause the calibration node to record an incorrect Z height.

## Project structure

```text
├── bashrc
├── Dockerfile
├── entrypoint.sh
├── imgs
│   └── img_system.png
├── LICENSE
├── README.md
├── ur5_algoritmos
│   ├── launch
│   │   ├── dibujo_completo.launch.py
│   │   ├── ur5_draw_v1.launch.py
│   │   └── ur5_view.launch.py
│   ├── package.xml
│   ├── resource
│   │   └── ur5_algoritmos
│   ├── setup.cfg
│   ├── setup.py
│   ├── test
│   │   ├── test_copyright.py
│   │   ├── test_flake8.py
│   │   └── test_pep257.py
│   └── ur5_algoritmos
│       ├── array_new_v8.py
│       ├── calibration_draw.py
│       ├── fk_functions.py
│       ├── ik_functions.py
│       ├── imagen_trajectory_new.py
│       ├── __init__.py
│       ├── kine_control_functions.py
│       ├── letter_trajectory_functions_img.py
│       ├── letter_trajectory_functions_v7.py
│       ├── markers.py
│       ├── move_draw_sub.py
│       ├── Playwrite_CU
│       │   ├── OFL.txt
│       │   ├── PlaywriteCU-VariableFont_wght.ttf
│       │   ├── README.txt
│       │   └── static
│       │       ├── PlaywriteCU-ExtraLight.ttf
│       │       ├── PlaywriteCU-Light.ttf
│       │       ├── PlaywriteCU-Regular.ttf
│       │       └── PlaywriteCU-Thin.ttf
│       └── QP_functions.py
└── ur_simulation_gazebo
    ├── CMakeLists.txt
    ├── config
    │   └── ur_controllers.yaml
    ├── launch
    │   ├── ur_sim_control.launch.py
    │   ├── ur_sim_lab201.launch.py
    │   ├── ur_sim_lab201.py
    │   └── ur_sim_moveit.launch.py
    ├── models
    │   ├── bidon
    │   │   ├── meshes
    │   │   │   ├── QR (1).png
    │   │   │   ├── QR.png
    │   │   │   ├── Untitled.mtl
    │   │   │   └── Untitled.obj
    │   │   ├── model.config
    │   │   └── model.sdf
    │   ├── surgery_table
    │   │   ├── meshes
    │   │   │   ├── surgery_table.dae
    │   │   │   └── surgery_table.stl
    │   │   ├── model.config
    │   │   └── model.sdf
    │   └── ur5_base
    │       ├── meshes
    │       │   ├── ur5_base.dae
    │       │   └── ur5_base.stl
    │       ├── model.config
    │       └── model.sdf
    ├── package.xml
    ├── test
    │   ├── test_common.py
    │   └── test_gazebo.py
    └── world
        └── lab_base_world_classic.sdf

```

## Troubleshooting

**`ModuleNotFoundError: No module named 'skimage'` after `pip3 install skimage`**
The PyPI package is named `scikit-image`, not `skimage` (the latter is an unrelated placeholder package that intentionally fails to install). Run `pip3 install --user scikit-image` instead.

**colcon warning: `AMENT_PREFIX_PATH ... doesn't exist`**
Harmless in most cases — it means a stale package path (e.g. from a deleted `install/` directory) is still cached in your shell environment. Open a fresh terminal, or `unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH` and re-source `/opt/ros/humble/setup.bash` followed by this workspace's `install/setup.bash`.

**Calibration never finishes / robot descends indefinitely in simulation**
Expected — see the Terminal 3 step above. `calibration_draw` only stops descending once it receives a `WrenchStamped` message with `force.z < -3` on `/force_torque_sensor_broadcaster/wrench`, and nothing in this simulation setup publishes that automatically.

## Features and roadmap

## References
