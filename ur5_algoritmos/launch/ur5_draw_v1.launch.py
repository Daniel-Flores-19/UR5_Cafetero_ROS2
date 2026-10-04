from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    image_node = Node(
        package = "ur5_algoritmos",
        executable = "img_trajectory",
        name = "img_trajectory_publisher",
        output = "screen",
    )

    robot_node = Node(
        package = "ur5_algoritmos",
        executable = "move_draw_sub",
        name = "ur5_kinecontrol_node",
        output = "screen",
    )

    text_node = Node(
        package = "ur5_algoritmos",
        executable = "letter_trajectory",
        name = "letter_trajectory_publisher",
        output = "screen",
    )

    return LaunchDescription([image_node, robot_node, text_node])
