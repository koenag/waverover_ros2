import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ----- Launch-time arguments (defaults mirror your declare_parameter defaults) -----
    image_endpoint =LaunchConfiguration("image_rect")
    camera_endpoint = LaunchConfiguration("camera_info")
    tag_family = LaunchConfiguration("family")
    tag_side_length = LaunchConfiguration("size")


    declare_args = [
        DeclareLaunchArgument('image_rect', default_value='/camera/image_raw'),
        DeclareLaunchArgument('camera_info', default_value='/camera/camera_info'),
        DeclareLaunchArgument('family', default_value='36h11'),
        DeclareLaunchArgument('size', default_value='0.092'),
    ]

    april_tag_detection_launch = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        remappings=[
            ('image_rect', image_endpoint),
            ('camera_info', camera_endpoint),
        ],
        parameters= [{
            'family': tag_family,
            'size': tag_side_length,
        }]
    )

    tag_localization_launch = Node(
        package='robot_localization',
        executable='tag_localization',
    )

    motor_control_launch = Node(
        package='robot_control',
        executable='motor_control',
    )

    movement_launch = Node(
        package='robot_control',
        executable='movement_control_simple',
    )

    camera_launch = os.path.join(
        get_package_share_directory('robot_vision_camera'),
        'launch',
        'robot_vision_camera.launch.py',
    )
    camera_launch_path = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(camera_launch),
        launch_arguments={'image_rectify': 'true'}.items(),
    )

    return LaunchDescription(declare_args + [camera_launch_path, april_tag_detection_launch, tag_localization_launch,motor_control_launch, movement_launch ])