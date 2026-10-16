
## Repository Structure

```text
waverover_ros2/
├── README.md
└── rubikpos2/
    └── robotcontrol/
        ├── setup.py
        └── robotcontrol/
            ├── motor_control.py
            ├── control_movement_simple.py
            └── control_movement_quintic.py
```

### Node Descriptions

- **`motor_control.py`**: Subscribes to motor commands from the movement controller and sends them to the robot's motors.

- **`control_movement_simple.py`**: Publishes motor commands to `motor_control.py`. This node originally used keyboard shortcuts, which have since been removed and replaced with an implicit kinematics model. It uses a turtle graphics approach to navigate the waypoint course through sequential rotations and straight-line movements.

- **`control_movement_quintic.py`**: Implements the quintic trajectory approach, generating quintic paths for the robot to follow. This node publishes motor commands but is not currently in use.

- **`setup.py`**: Defines the Python package configuration and registers the ROS 2 executable nodes.
