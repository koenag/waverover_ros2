from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

import os

def generate_launch_description():
    # ----- Launch-time arguments (defaults mirror your declare_parameter defaults) -----
    frame_rate = LaunchConfiguration('frame_rate')
    width = LaunchConfiguration('width')
    height = LaunchConfiguration('height')
    input_format = LaunchConfiguration('input_format')
    output_format = LaunchConfiguration('output_format')
    topic_name = LaunchConfiguration('topic_name')
    image_compress = LaunchConfiguration('image_compress')
    image_rectify = LaunchConfiguration('image_rectify')
    camera_parameter_path = LaunchConfiguration('camera_parameter_path')
    jpeg_quality = LaunchConfiguration('jpeg_quality')
    max_threads = LaunchConfiguration('max_threads')

    declare_args = [
        DeclareLaunchArgument('frame_rate', default_value='10'),
        DeclareLaunchArgument('width', default_value='1280'),
        DeclareLaunchArgument('height', default_value='720'),
        DeclareLaunchArgument('input_format', default_value='NV12'),
        DeclareLaunchArgument('output_format', default_value='BGR'),
        DeclareLaunchArgument('topic_name', default_value='camera/image_raw'),
        DeclareLaunchArgument('image_compress', default_value='true'),
        DeclareLaunchArgument('image_rectify', default_value='false'),
        DeclareLaunchArgument(
            'camera_parameter_path',
            default_value=PathJoinSubstitution([
                FindPackageShare('robot_vision_camera'),
                'config',
                'camera_parameter.yaml'
            ])
        ),
        DeclareLaunchArgument('jpeg_quality', default_value='90'),
        DeclareLaunchArgument('max_threads', default_value='1'),
    ]

    # ----- Robot Vision Camera node -----
    # Replace package/executable with your actual ones if different.
    robot_vision_node = Node(
        package='robot_vision_camera',
        executable='robot_vision_camera_node', 
        name='robot_vision_camera',
        output='screen',
        # If you need to set env (e.g., GST_DEBUG), uncomment:
        # env={'GST_DEBUG': '2'},
        parameters=[{
            'frame_rate': frame_rate,
            'width': width,
            'height': height,
            'input_format': input_format,
            'output_format': output_format,
            'topic_name': topic_name,
            'image_compress': image_compress,
            'image_rectify': image_rectify,
            'camera_parameter_path': camera_parameter_path,
            'jpeg_quality': jpeg_quality,
            'max_threads': max_threads,
        }],
        # Example remaps (optional): publish compressed under <topic_name>/compressed by your code already
        # remappings=[('camera/image_raw', LaunchConfiguration('topic_name'))],
    )

    # ----- Foxglove Bridge Launch -----
    # Launch foxglove_bridge node directly since XML launch files aren't easily supported
    foxglove_launch = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': 8765,
            'address': '0.0.0.0',
            'tls': False,
            'certfile': '',
            'keyfile': '',
            'topic_whitelist': ['.*'],
            'param_whitelist': ['.*'],
            'service_whitelist': ['.*'],
            'client_topic_whitelist': ['.*'],
            'min_qos_depth': 1,
            'max_qos_depth': 25,
            'num_threads': 0,
            'use_compression': False,
        }]
    )

    return LaunchDescription(declare_args + [robot_vision_node, foxglove_launch])