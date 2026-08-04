from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    image_node = Node(
        package = "ur5_algoritmos",
        executable = "imagen_trajectory_new",
        name = "imagen_trajectory_publisher",
        output = "screen",
    )

    robot_node = Node(
        package = "ur5_algoritmos",
        executable = "move_draw_new",
        name = "ur5_kinecontrol_node",
        output = "screen",
    )

    text_node = Node(
        package = "ur5_algoritmos",
        executable = "array_new_v8",
        name = "letter_trajectory_publisher",
        output = "screen",
    )

    return LaunchDescription([image_node, robot_node, text_node])
