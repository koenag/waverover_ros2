
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32MultiArray


# Robot calibration: [L, R] -> [linear velocity, angular velocity]
CALIBRATION_MATRIX = np.array([
    [0.565034, 0.733764],
    [-4.08242, 3.82298]
])
CALIBRATION_INVERSE = np.linalg.inv(CALIBRATION_MATRIX)

MAX_MOTOR_COMMAND = 0.5
DRIVE_SPEED = 0.30
TURN_SPEED = 3.0

POSE_TIMEOUT = 1.0
SEARCH_TURN_SPEED = 3.0


def wrap_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    return math.pi if abs(wrapped + math.pi) < 0.01 else wrapped


def parse_waypoints(filename):
    with open(filename, "r") as file:
        return [
            tuple(map(float, line.split(",")))
            for line in file
            if line.strip()
        ]


class WaypointController(Node):

    def __init__(self):
        super().__init__("keyboard_controller_node")

        self.publisher = self.create_publisher(
            Float32MultiArray, "motor_commands", 10
        )
        self.create_subscription(
            Pose2D, "/robot_pose", self.pose_callback, 10
        )

        # Latest robot pose from AprilTag localization
        self.robot_x = 0.69
        self.robot_y = 0.4
        self.robot_theta = 0.0
        self.last_pose_time = 0.0

        # Motor state
        self.current_L = 0.0
        self.current_R = 0.0
        self.calibration_fault = False

        # Planned actions and current progress
        waypoints = parse_waypoints("./waypoint_2.txt")
        self.actions = self.build_actions(waypoints)
        self.action_index = 0
        self.action_start_pose = None

        self.log_count = 0

        self.create_timer(0.10, self.update)
        self.get_logger().info("Waypoint controller started")


    def pose_callback(self, msg):
        """Store the latest localized pose."""
        self.robot_x = msg.x
        self.robot_y = msg.y
        self.robot_theta = msg.theta
        self.last_pose_time = time.monotonic()


    def localization_available(self):
        """Check whether a recent localization update exists."""
        return (
            self.last_pose_time != 0.0
            and time.monotonic() - self.last_pose_time <= POSE_TIMEOUT
        )


    def set_motor_speeds(self, v, omega):
        """Convert desired velocities to calibrated motor commands."""
        L, R = CALIBRATION_INVERSE @ np.array([v, omega])

        if not np.all(np.isfinite([L, R])) or max(abs(L), abs(R)) > MAX_MOTOR_COMMAND:
            self.current_L = 0.0
            self.current_R = 0.0
            self.calibration_fault = True

            self.get_logger().error(
                f"Unachievable command: v={v:.4f}, omega={omega:.4f}, "
                f"L={L:.4f}, R={R:.4f}. Stopping."
            )
            return

        self.current_L = float(L)
        self.current_R = float(R)


    def build_actions(self, waypoints):
        """Convert waypoints into turn, drive, and final-turn actions."""
        actions = []

        for i in range(len(waypoints) - 1):
            x0, y0, theta0 = waypoints[i]
            x1, y1, theta1 = waypoints[i + 1]

            dx = x1 - x0
            dy = y1 - y0
            distance = math.hypot(dx, dy)

            drive_heading = (
                math.atan2(dy, dx) if distance > 1e-6 else theta0
            )

            initial_turn = wrap_angle(drive_heading - theta0)
            final_turn = wrap_angle(theta1 - drive_heading)

            if abs(initial_turn) > 0.02:
                actions.append((
                    0.0,
                    math.copysign(TURN_SPEED, initial_turn),
                    abs(initial_turn) / TURN_SPEED,
                    f"Segment {i}: Initial turn"
                ))

            if distance > 1e-6:
                actions.append((
                    DRIVE_SPEED,
                    0.0,
                    distance / DRIVE_SPEED,
                    f"Segment {i}: Drive straight"
                ))

            if abs(final_turn) > 0.02:
                actions.append((
                    0.0,
                    math.copysign(TURN_SPEED, final_turn),
                    abs(final_turn) / TURN_SPEED,
                    f"Segment {i}: Final turn"
                ))

        return actions


    def get_action_progress(self, v, omega):
        """Calculate action progress using the localized robot pose."""
        start_x, start_y, start_theta = self.action_start_pose

        if abs(v) > 1e-6:
            dx = self.robot_x - start_x
            dy = self.robot_y - start_y

            distance_traveled = (
                dx * math.cos(start_theta)
                + dy * math.sin(start_theta)
            )
            return max(0.0, distance_traveled / v)

        if abs(omega) > 1e-6:
            angle_turned = wrap_angle(self.robot_theta - start_theta)
            return max(0.0, angle_turned / omega)

        return 0.0


    def complete_action(self, description):
        """Advance to the next planned action."""
        self.get_logger().info(f"ACTION COMPLETED: {description}")

        self.action_index += 1
        self.action_start_pose = None

        if self.action_index >= len(self.actions):
            self.current_L = 0.0
            self.current_R = 0.0
            self.get_logger().info("ALL WAYPOINTS COMPLETED")


    def control_step(self):
        """Execute the current action using localization feedback."""
        if self.calibration_fault or self.action_index >= len(self.actions):
            self.current_L = 0.0
            self.current_R = 0.0
            return

        # Search for an AprilTag if localization is unavailable.
        if not self.localization_available():
            self.get_logger().info("AprilTag lost. Rotating to search.")
            self.set_motor_speeds(0.0, SEARCH_TURN_SPEED)
            return

        v, omega, duration, description = self.actions[self.action_index]

        # Record the pose at the beginning of each action.
        if self.action_start_pose is None:
            self.action_start_pose = (
                self.robot_x,
                self.robot_y,
                self.robot_theta
            )

        progress = self.get_action_progress(v, omega)

        # Apply calibrated motor commands.
        self.set_motor_speeds(v, omega)

        if self.calibration_fault:
            return

        self.log_count += 1
        if self.log_count % 2 == 0:
            self.get_logger().info(
                f"{description}: t={progress:.2f}/{duration:.2f}, "
                f"L={self.current_L:.4f}, R={self.current_R:.4f}"
            )

        if progress >= duration:
            self.complete_action(description)


    def publish_motor_commands(self):
        msg = Float32MultiArray()
        msg.data = [self.current_L, self.current_R]
        self.publisher.publish(msg)


    def update(self):
        self.control_step()
        self.publish_motor_commands()


    def destroy_node(self):
        self.current_L = 0.0
        self.current_R = 0.0
        self.publish_motor_commands()
        self.get_logger().info("Stop command sent")
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
