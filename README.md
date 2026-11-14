
## Repository Structure

```text
waverover_ros2/
├── README.md
├── calibrate_camera.py
├── plot_trajectory.py
├── waypoint_1.txt
├── waypoint_2.txt
└── rubikpi_ros2/
    ├── README.md
    ├── robot_control/
    │   ├── package.xml
    │   ├── setup.py
    │   ├── setup.cfg
    │   ├── resource/
    │   │   └── robot_control
    │   ├── config/
    │   │   └── apriltag_map.yaml              # Known AprilTag positions and orientations
    │   ├── launch/
    │   │   └── robot_teleop_launch.py
    │   └── robot_control/
    │       ├── __init__.py
    │       ├── motor_control.py               # Motor commands, serial communication
    │       ├── movement_control_simple.py     # Turtle graphics waypoint navigation
    │       ├── movement_control_quintic.py    # Quintic trajectory controller
    │       ├── tag_localization.py            # AprilTag locations in camera coordinate system transforms to robot pose in world 
    │       ├── slam_node.py                   # Slam node
    │       ├── ekf_slam.py                    # Performs EKF correction
    coordinate system
    │       └── teleop_example.py
    ├── robot_launch/
    │   ├── package.xml
    │   ├── setup.py
    │   ├── setup.cfg
    │   ├── resource/
    │   │   └── robot_launch
    │   └── launch/
    │       └── robot_control_movement.launch.py  # Launches camera, detection,
    │                                            # localization, and motion control
    ├── robot_vision/
    │   ├── package.xml
    │   ├── CMakeLists.txt
    │   ├── config/
    │   │   └── camera_parameter.yaml          # Camera intrinsics and distortion
    │   ├── launch/
    │   │   └── robot_vision_camera.launch.py
    │   ├── include/
    │   │   └── robot_vision_camera/
    │   │       └── robot_vision_camera.h
    │   └── src/
    │       └── robot_vision_camera.cpp
    └── apriltag_ros/
        ├── README.md
        ├── package.xml
        ├── CMakeLists.txt
        ├── cfg/
        │   └── tags_36h11.yaml                # Detector settings and tag definitions
        ├── launch/
        │   ├── apriltag_launch.py
        │   └── camera_36h11.launch.yml
        └── src/
            ├── AprilTagNode.cpp
            ├── conversion.cpp
            ├── pose_estimation.cpp
            ├── pose_estimation.hpp
            ├── tag_functions.cpp
            └── tag_functions.hpp
```
