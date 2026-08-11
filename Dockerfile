# Import ROS distro
FROM osrf/ros:humble-desktop-full

# Installing minor pkgs
RUN apt-get update \
  && apt-get install -y \
  nano \
  vim \
  && rm -rf /var/lib/apt/lists/*

# Creating and installing controllers and repos
RUN mkdir -p ros2_ws/src \
    && cd ros2_ws/src/ \
    && apt-get update \
    && apt-get install -y git python3-rosdep python3-vcstool \
    && apt-get install -y ros-humble-forward-command-controller \
    && apt-get install -y ros-humble-ros2-controllers \
    && git clone https://github.com/Daniel-Flores-19/UR5_Cafetero_ROS2.git \
    && git clone -b humble https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation.git ur_simulation_gz \
    && printf "repositories:\n  Universal_Robots_ROS2_Description:\n    type: git\n    url: https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git\n    version: humble\n" > ur_description_only.repos \
    && vcs import . < ur_description_only.repos \
    && for i in 1 2 3 4 5; do rosdep update && break || sleep 5; done \
    && rosdep install --ignore-src --from-paths . -y \
    && rm -rf /var/lib/apt/lists/*

# Installing more dependencies
RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip3 install osqp scikit-image \
    && rm -rf /var/lib/apt/lists/*

# Creating a sudo user
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config

# Enabling sudo and autocompletion
RUN apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/lists/*

# Copying entrypoint and bashrc files
COPY entrypoint.sh /entrypoint.sh
COPY bashrc /home/${USERNAME}/.bashrc

ENTRYPOINT [ "/bin/bash", "/entrypoint.sh" ]
# Enter this docker as root
USER root

CMD ["bash"]