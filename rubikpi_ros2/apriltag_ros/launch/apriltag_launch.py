#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    
    # Declare launch arguments
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Input image topic'
    )
    
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/camera_info',
        description='Camera info topic'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='tags_36h11.yaml',
        description='AprilTag configuration file name'
    )
    
    # Get configuration file path
    config_file_path = PathJoinSubstitution([
        FindPackageShare('apriltag_ros'),
        'cfg',
        LaunchConfiguration('config_file')
    ])
    
    # AprilTag node
    apriltag_node = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_node',
        remappings=[
            ('image_rect', LaunchConfiguration('image_topic')),
            ('camera_info', LaunchConfiguration('camera_info_topic'))
        ],
        parameters=[config_file_path],
        output='screen'
    )
    
    return LaunchDescription([
        image_topic_arg,
        camera_info_topic_arg,
        config_file_arg,
        apriltag_node
    ])