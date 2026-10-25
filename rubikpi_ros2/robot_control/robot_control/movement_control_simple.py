import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32MultiArray

CALIBRATION_MATRIX = np.array([
    [0.565034, 0.733764],
    [-4.08242, 3.82298]
])
CALIBRATION_INVERSE = np.linalg.inv(CALIBRATION_MATRIX)

MAX_MOTOR_COMMAND = 0.5
DRIVE_SPEED = 0.12       # m/s
TURN_SPEED = 1.5        # rad/s
POSITION_TOLERANCE = 0.07  # m
HEADING_TOLERANCE = 0.12   # rad
POSE_TIMEOUT = 1.5      # seconds
SEARCH_TURN_SPEED = 1.5
SEARCH_ROTATE_TIME = 0.25
SEARCH_PAUSE_TIME = 1.25


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def parse_waypoints(filename):
    with open(filename) as file:
        return [tuple(map(float, line.split(','))) for line in file if line.strip()]


class WaypointController(Node):
    def __init__(self):
        super().__init__('waypoint_controller_node')
        self.publisher = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.create_subscription(Pose2D, '/robot_pose', self.pose_callback, 10)

        self.waypoints = parse_waypoints('./waypoint_2.txt')
        self.index = 1  # First waypoint is the starting position.
        self.state = 'TURN'
        self.pose = None
        self.last_pose_time = 0.0
        self.search_start = None

        self.create_timer(0.1, self.update)
        self.get_logger().info('Waypoint controller started')

    def pose_callback(self, msg):
        self.pose = (msg.x, msg.y, msg.theta)
        self.last_pose_time = time.monotonic()

    def send_velocity(self, v, omega):
        L, R = CALIBRATION_INVERSE @ np.array([v, omega])
        # Preserve turn direction and curvature if a command exceeds the limit.
        scale = max(1.0, abs(L) / MAX_MOTOR_COMMAND, abs(R) / MAX_MOTOR_COMMAND)
        msg = Float32MultiArray()
        msg.data = [float(L / scale), float(R / scale)]
        self.publisher.publish(msg)

    def update(self):
        now = time.monotonic()

        if self.index >= len(self.waypoints):
            self.send_velocity(0.0, 0.0)
            return

        # No recent AprilTag pose: rotate, then pause to allow detection.
        if self.pose is None or now - self.last_pose_time > POSE_TIMEOUT:
            if self.search_start is None:
                self.search_start = now
            cycle = SEARCH_ROTATE_TIME + SEARCH_PAUSE_TIME
            if (now - self.search_start) % cycle < SEARCH_ROTATE_TIME:
                self.send_velocity(0.0, SEARCH_TURN_SPEED)
            else:
                self.send_velocity(0.0, 0.0)
            return

        self.search_start = None
        x, y, theta = self.pose
        goal_x, goal_y, goal_theta = self.waypoints[self.index]
        dx, dy = goal_x - x, goal_y - y
        distance_to_goal = math.hypot(dx, dy)
        heading_to_goal = math.atan2(dy, dx)
        heading_error = wrap_angle(heading_to_goal - theta) # turning to get to next waypoint
        final_error = wrap_angle(goal_theta - theta) # turning to get to final orientation at waypoint

        if self.state == 'TURN':
            if distance_to_goal <= POSITION_TOLERANCE:
                self.state = 'FINAL_TURN'
            elif abs(heading_error) > HEADING_TOLERANCE:
                self.send_velocity(0.0, math.copysign(TURN_SPEED, heading_error))
                return
            else:
                self.state = 'DRIVE'

        if self.state == 'DRIVE':
            if distance_to_goal <= POSITION_TOLERANCE:
                self.state = 'FINAL_TURN'
            elif abs(heading_error) > HEADING_TOLERANCE:
                self.state = 'TURN'
                self.send_velocity(0.0, 0.0)
                return
            else:
                self.send_velocity(DRIVE_SPEED, 0.0)
                return

        if self.state == 'FINAL_TURN':
            if abs(final_error) > HEADING_TOLERANCE:
                self.send_velocity(0.0, math.copysign(TURN_SPEED, final_error))
                return

            self.get_logger().info(f'Reached waypoint {self.index}')
            self.index += 1
            self.state = 'TURN'
            self.send_velocity(0.0, 0.0)
            return

    def destroy_node(self):
        self.send_velocity(0.0, 0.0)
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


if __name__ == '__main__':
    main()
