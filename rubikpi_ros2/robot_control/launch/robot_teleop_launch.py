from launch import LaunchDescription
from launch.actions import RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    motor_node = Node(
        package='robot_control',
        executable='motor_control',
        name='motor_controller_node',
        output='screen',
        emulate_tty=True,
        parameters=[],
    )

    keyboard_node = Node(
        package='robot_control',
        executable='keyboard_control',
        name='keyboard_controller_node',
        output='screen',
        emulate_tty=True,
        parameters=[],
    )

    # If the keyboard node exits (ESC), bring the whole system down.
    shutdown_when_keyboard_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=keyboard_node,
            on_exit=[Shutdown(reason='Keyboard controller exited')]
        )
    )

    # If the motor controller exits for any reason, also bring everything down.
    shutdown_when_motor_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=motor_node,
            on_exit=[Shutdown(reason='Motor controller exited')]
        )
    )

    return LaunchDescription([
        motor_node,
        keyboard_node,
        shutdown_when_keyboard_exits,
        shutdown_when_motor_exits,
    ])
