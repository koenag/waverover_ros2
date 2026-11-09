import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D, TwistStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


# Motion settings
CALIBRATION_MATRIX = np.array([
    [0.565034, 0.733764],
    [-4.08242, 3.82298]
])
MAX_MOTOR_COMMAND = 0.5
DRIVE_SPEED = 0.2
TURN_SPEED = 3.0
POSITION_TOLERANCE = 0.1
HEADING_TOLERANCE = 0.2
MAX_DRIVE_SEGMENT = 0.20

CALLBACK_FREQUENCY = 0.05   # 20 Hz command heartbeat

# EKF-SLAM defines the starting robot pose as (0, 0, 0).
# Only use a nonzero offset if the waypoint file explicitly requires it.
OFFSET = (0.0, 0.0, 0.0)


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class MovementController(Node):
    def __init__(self):
        '''
        Publishes L,R to motor_commands
        Publishes the effective [v, w] to /slam/cmd_velocity (every tick)
        Subscribes to /slam/pose (live estimate) and /slam/segment_pose
        (once per stop, after slam's settle + observation window)
        '''
        super().__init__('movement_controller_node')

        # Motors take float32 [left, right]
        self.publisher = self.create_publisher(
            Float32MultiArray, 'motor_commands', 10)

        # Slam integrates the effective v, w AFTER motor saturation.
        self.slam_publisher = self.create_publisher(
            TwistStamped, '/slam/cmd_velocity', 10)

        # The live pose gives the starting pose. Navigation waits for the
        # segment pose, which arrives even if no tag was seen.
        self.create_subscription(Pose2D, '/slam/pose', self.update_pose, 10)
        self.create_subscription(
            Pose2D, '/slam/segment_pose', self.update_segment_pose, 10)

        with open('./waypoint_3.txt') as file:
            self.waypoints = [
                (x + OFFSET[0], y + OFFSET[1], wrap_angle(theta + OFFSET[2]))
                for line in file if line.strip()
                for x, y, theta in [map(float, line.split(','))]
            ]

        self.waypoint_index = 1  # Waypoint 0 is the starting position

        self.pose = None             # latest pose from slam (x, y, theta)
        self.segment_pose = None     # slam's reply to the last burst, until used
        self.waiting_slam = False    # True from the end of a burst until slam replies

        # (requested linear, requested angular, start time, duration, description)
        self.movement = None

        # Once within POSITION_TOLERANCE of a waypoint, only fix the heading,
        # so a later EKF correction can't send the robot back to revisit it.
        self.position_reached = False

        self.create_timer(CALLBACK_FREQUENCY, self.move)

        self.get_logger().info(
            f'[START] {len(self.waypoints)} waypoints loaded | '
            f'Drive speed={DRIVE_SPEED:.2f}m/s | '
            f'Turn speed={TURN_SPEED:.2f}rad/s'
        )

    def update_pose(self, msg):
        '''Live estimate, including prediction-only updates.'''
        self.pose = (msg.x, msg.y, msg.theta)

    def update_segment_pose(self, msg):
        '''Slam's reply to a burst. Ignored unless we are waiting for one.'''
        if not self.waiting_slam:
            return
        self.segment_pose = (msg.x, msg.y, msg.theta)

    def command(self, linear_speed, angular_speed):
        '''Send L,R to the motors and the effective (v, w) to slam.'''
        left, right = np.linalg.solve(
            CALIBRATION_MATRIX, [linear_speed, angular_speed])

        scale = max(
            1.0,
            abs(left) / MAX_MOTOR_COMMAND,
            abs(right) / MAX_MOTOR_COMMAND
        )

        self.publisher.publish(Float32MultiArray(
            data=[float(left / scale), float(right / scale)]))

        # Tell slam what the motors actually do, not what was requested.
        actual_linear, actual_angular = linear_speed / scale, angular_speed / scale
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.twist.linear.x = float(actual_linear)
        twist.twist.angular.z = float(actual_angular)
        self.slam_publisher.publish(twist)

        return actual_linear, actual_angular

    def create_movement(self, linear_speed, angular_speed, amount, description):
        # Forget any earlier reply before the new command goes out.
        self.segment_pose = None

        actual_linear, actual_angular = self.command(linear_speed, angular_speed)

        actual_speed = abs(actual_linear if linear_speed != 0 else actual_angular)
        if actual_speed < 1e-6:
            self.get_logger().error('Requested movement produced zero effective speed')
            self.command(0.0, 0.0)
            return

        duration = amount / actual_speed
        self.movement = (linear_speed, angular_speed, time.monotonic(),
                         duration, description)

        self.get_logger().info(
            f'[MOVE START] {description} | '
            f'Amount={amount:.3f}{"m" if linear_speed != 0 else "rad"} | '
            f'Duration={duration:.2f}s | '
            f'v={actual_linear:+.3f}m/s | w={actual_angular:+.3f}rad/s'
        )

    def move(self):
        '''
        Burst of open-loop movement, then stop and wait for slam's segment
        pose, then plan the next burst from that pose.
        '''
        now = time.monotonic()

        # In a burst: repeat the command, then stop at the end
        if self.movement is not None:
            linear, angular, start, duration, description = self.movement

            if now - start < duration:
                self.command(linear, angular)
                return

            self.command(0.0, 0.0)
            self.movement = None
            self.waiting_slam = True
            self.get_logger().info(
                f'[MOVE END] {description} | waiting for slam segment pose')
            return

        # Stopped after a burst: keep sending zero heartbeats until slam's
        # settle + observation window ends. Slam replies even with no tag.
        if self.waiting_slam:
            self.command(0.0, 0.0)
            if self.segment_pose is None:
                return
            self.pose = self.segment_pose
            self.segment_pose = None
            self.waiting_slam = False

        # Wait for slam's starting pose instead of requiring a first tag.
        if self.pose is None:
            self.command(0.0, 0.0)
            return

        x, y, theta = self.pose
        goal_x, goal_y, goal_theta = self.waypoints[self.waypoint_index]

        dx = goal_x - x
        dy = goal_y - y
        distance_remaining = math.hypot(dx, dy)
        heading_error = wrap_angle(math.atan2(dy, dx) - theta)
        final_heading_error = wrap_angle(goal_theta - theta)

        self.get_logger().info(
            f'[POSE] WP {self.waypoint_index} | '
            f'Robot=({x:.3f}, {y:.3f}, {math.degrees(theta):+.1f}deg) | '
            f'Target=({goal_x:.3f}, {goal_y:.3f}, {math.degrees(goal_theta):+.1f}deg) | '
            f'dist={distance_remaining:.3f}m | '
            f'heading err={math.degrees(heading_error):+.1f}deg | '
            f'final err={math.degrees(final_heading_error):+.1f}deg'
        )

        if distance_remaining <= POSITION_TOLERANCE:
            self.position_reached = True

        if self.position_reached:
            if abs(final_heading_error) <= HEADING_TOLERANCE:
                self.get_logger().info(f'[WAYPOINT REACHED] {self.waypoint_index}')

                # Wrap around to keep traversing the square.
                self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
                self.position_reached = False
            else:
                self.create_movement(
                    0.0, math.copysign(TURN_SPEED, final_heading_error),
                    abs(final_heading_error), 'FINAL TURN')

        elif abs(heading_error) > HEADING_TOLERANCE:
            self.create_movement(
                0.0, math.copysign(TURN_SPEED, heading_error),
                abs(heading_error), 'TURN')

        else:
            self.create_movement(
                DRIVE_SPEED, 0.0,
                min(distance_remaining, MAX_DRIVE_SEGMENT), 'DRIVE')


def main(args=None):
    rclpy.init(args=args)
    node = MovementController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.command(0.0, 0.0)  # fails if Ctrl-C already invalidated the context
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()