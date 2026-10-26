import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32MultiArray


# Motion settings
CALIBRATION = np.array([
    [0.565034, 0.733764],
    [-4.08242, 3.82298]
])
MAX_MOTOR_COMMAND = 0.5
DRIVE_SPEED = 0.2
TURN_SPEED = 3.0
POSITION_TOLERANCE = 0.10
HEADING_TOLERANCE = 0.2
POSE_TIMEOUT = 3.0

# Localization search settings
SEARCH_INITIAL_PAUSE = 2.0
SEARCH_ROTATION_TIME = 0.10
SEARCH_PAUSE = 3.0

CONTROL_PERIOD = 0.05


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class WaypointController(Node):
    def __init__(self):
        super().__init__('waypoint_controller_node')

        self.publisher = self.create_publisher(
            Float32MultiArray, 'motor_commands', 10)
        self.create_subscription(
            Pose2D, '/robot_pose', self.update_pose, 10)

        with open('./waypoint_2.txt') as file:
            self.waypoints = [
                tuple(map(float, line.split(',')))
                for line in file if line.strip()
            ]

        self.waypoint_index = 1  # Waypoint 0 is the starting position
        self.pose = None
        self.pose_time = 0.0
        self.used_pose_time = 0.0

        # (linear speed, angular speed, end time, movement name)
        self.movement = None

        # (search phase, phase start time)
        self.search = None

        self.create_timer(CONTROL_PERIOD, self.update)
        self.get_logger().info(
            f'Controller started with {len(self.waypoints)} waypoints')

    def update_pose(self, message):
        self.pose = (message.x, message.y, message.theta)
        self.pose_time = time.monotonic()

    def command(self, linear_speed, angular_speed):
        left, right = np.linalg.solve(
            CALIBRATION, [linear_speed, angular_speed])

        scale = max(
            1.0,
            abs(left) / MAX_MOTOR_COMMAND,
            abs(right) / MAX_MOTOR_COMMAND
        )

        self.publisher.publish(Float32MultiArray(
            data=[float(left / scale), float(right / scale)]))

        return linear_speed / scale, angular_speed / scale

    def start_move(self, linear_speed, angular_speed, amount, now, name):
        actual_linear, actual_angular = self.command(
            linear_speed, angular_speed)

        actual_speed = abs(
            actual_linear if linear_speed != 0 else actual_angular)

        duration = amount / max(actual_speed, 1e-6)
        self.movement = (
            linear_speed, angular_speed, now + duration, name)

        self.get_logger().info(
            f'{name}: amount={amount:.2f}, duration={duration:.2f}s')

    def update(self):
        now = time.monotonic()

        # Finish the current movement before using another pose.
        if self.movement:
            linear_speed, angular_speed, end_time, name = self.movement

            if now < end_time:
                self.command(linear_speed, angular_speed)
                return

            self.command(0.0, 0.0)
            self.movement = None
            self.used_pose_time = self.pose_time
            self.search = None
            self.get_logger().info(f'Finished {name}; waiting for a new pose')
            return

        # All waypoints completed.
        if self.waypoint_index >= len(self.waypoints):
            self.command(0.0, 0.0)
            return

        # Only plan using a pose received after the previous movement ended.
        #if its a fresh pose (position ) and youve finished the segment
        if self.pose is not None and self.pose_time > self.used_pose_time:
            self.used_pose_time = self.pose_time
            self.search = None

            x, y, heading = self.pose
            goal_x, goal_y, goal_heading = self.waypoints[self.waypoint_index]

            dx = goal_x - x
            dy = goal_y - y
            distance_remaining = math.hypot(dx, dy)

            heading_error = wrap_angle(math.atan2(dy, dx) - heading)
            final_heading_error = wrap_angle(goal_heading - heading)

            if distance_remaining <= POSITION_TOLERANCE:
                if abs(final_heading_error) <= HEADING_TOLERANCE:
                    self.get_logger().info(
                        f'Reached waypoint {self.waypoint_index}: '
                        f'{self.waypoints[self.waypoint_index]}')
                    self.waypoint_index += 1
                else:
                    # if you haven't finished the final turn, continue turning along that error till you finish
                    self.start_move(
                        0.0,
                        math.copysign(TURN_SPEED, final_heading_error), 
                        abs(final_heading_error), now, 'FINAL TURN')
            # if you haven't reached the target yet and you're not facing the correct direction turn to face the correct direction
            elif abs(heading_error) > HEADING_TOLERANCE:
                self.start_move(
                    0.0,
                    math.copysign(TURN_SPEED, heading_error),
                    abs(heading_error), now, 'TURN')
            # if you're facing the right way but not in the right position just go straight
            else:
                # Slow down when approaching the waypoint.
                speed = min(DRIVE_SPEED, max(0.08, 0.8 * distance_remaining))
                self.start_move(
                    speed, 0.0, distance_remaining, now, 'DRIVE')

            return

        # no new pose, don't move for a split second so you can decide how to search
        self.command(0.0, 0.0)

        # if you can't see an april right now and haven't seen one in POSE_TIMEOUT seconds
        if self.pose is None or now - self.pose_time > POSE_TIMEOUT:
            if self.search is None:
                self.search = ('INITIAL_PAUSE', now)
                self.get_logger().warning('Localization lost; searching')

            phase, start_time = self.search
            elapsed = now - start_time

            if phase == 'INITIAL_PAUSE' and elapsed >= SEARCH_INITIAL_PAUSE:
                self.search = ('ROTATE', now)
            elif phase == 'ROTATE' and elapsed >= SEARCH_ROTATION_TIME:
                self.search = ('PAUSE', now)
            elif phase == 'PAUSE' and elapsed >= SEARCH_PAUSE:
                self.search = ('ROTATE', now)

            if self.search[0] == 'ROTATE':
                self.command(0.0, TURN_SPEED)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.command(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
