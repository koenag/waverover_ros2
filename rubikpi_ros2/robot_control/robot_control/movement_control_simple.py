import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray


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

CALLBACK_FREQUENCY = 0.05

# Must match the EKF's initial state (zeros) and the frame the waypoints are in.
OFFSET = (0.0, 0.0, 0.0)

STARTUP_WAIT = 3.0   # seconds to let slam come up before the first burst


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class MovementController(Node):
    def __init__(self):
        '''
        Publishes L,R to motor_commands
        Publishes [v, w, dt, stop_time, end_of_delay] to robot_movement_vw
        Subscribes to verified_pose_update from slam
        '''
        super().__init__('movement_controller_node')

        # Motors take float32 [left, right]
        self.publisher = self.create_publisher(
            Float32MultiArray, 'motor_commands', 10)

        # Slam topics are float64 (absolute ROS times don't fit in float32)
        self.slam_publisher = self.create_publisher(
            Float64MultiArray, 'robot_movement_vw', 10)
        self.create_subscription(
            Float64MultiArray, 'verified_pose_update', self.update_pose, 10)

        with open('./waypoint_2.txt') as file:
            self.waypoints = [
                (x + OFFSET[0], y + OFFSET[1], theta + OFFSET[2])
                for line in file if line.strip()
                for x, y, theta in [map(float, line.split(','))]
            ]

        self.waypoint_index = 1  # Waypoint 0 is the starting position

        self.pose = None                   # latest pose from slam; None until slam publishes
        self.landmarks = {}                # latest landmarks from slam
        self.pose_time = time.monotonic()  # when self.pose was last set
        self.used_pose_time = 0.0          # pose_time the planner last used

        # Wait for slam to come up before the first burst
        self.pause_until = time.monotonic() + STARTUP_WAIT

        # (actual linear, actual angular, start time, duration, description)
        self.movement = None

        # Once within POSITION_TOLERANCE of a waypoint, only fix the heading,
        # so a later EKF correction can't send the robot back to revisit it.
        self.position_reached = False

        # Slam handshake
        self.burst_count = 0         # bursts sent to slam (for logs only)
        self.waiting_slam = False    # True from the end of a burst until slam replies
        self.last_wait_log = 0.0

        self.create_timer(CALLBACK_FREQUENCY, self.move)

        self.get_logger().info(
            f'[START] {len(self.waypoints)} waypoints loaded | '
            f'Drive speed={DRIVE_SPEED:.2f}m/s | '
            f'Turn speed={TURN_SPEED:.2f}rad/s'
        )

    def ros_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def update_pose(self, msg):
        '''
        Slam's message: [ok, x, y, theta, n, id, mx, my, ...]
        The first one is the starting pose. After that slam sends exactly one
        per burst, so a message while waiting is the reply to that burst.
        '''
        d = msg.data
        ok = d[0]

        if self.pose is None:
            self.pose = (d[1], d[2], d[3])
            self.pose_time = time.monotonic()
            self.get_logger().info(
                f'[START POSE] from slam: ({d[1]:.3f}, {d[2]:.3f}, '
                f'{math.degrees(d[3]):+.1f}deg)')
            return

        if not self.waiting_slam:
            return  # not expecting anything from slam

        if ok:
            self.pose = (d[1], d[2], d[3])
        else:
            self.get_logger().warning('slam update failed; keeping last pose')

        n = int(d[4])
        self.landmarks = {int(d[5 + 3*k]): (d[6 + 3*k], d[7 + 3*k]) for k in range(n)}
        self.waiting_slam = False            # lets the next burst start
        self.pose_time = time.monotonic()    # marks the pose as new for the planner
        self.get_logger().info(
            f'[SLAM] burst {self.burst_count} pose=({self.pose[0]:.3f}, '
            f'{self.pose[1]:.3f}, {math.degrees(self.pose[2]):+.1f}deg) tags={n}')

    def command(self, linear_speed, angular_speed):
        '''Send L,R to the motors. Returns the (v, w) actually commanded after saturation.'''
        left, right = np.linalg.solve(
            CALIBRATION_MATRIX, [linear_speed, angular_speed])

        scale = max(
            1.0,
            abs(left) / MAX_MOTOR_COMMAND,
            abs(right) / MAX_MOTOR_COMMAND
        )

        self.publisher.publish(Float32MultiArray(
            data=[float(left / scale), float(right / scale)]))

        return linear_speed / scale, angular_speed / scale

    def create_movement(self, linear_speed, angular_speed, amount, description):
        actual_linear, actual_angular = self.command(linear_speed, angular_speed)

        actual_speed = abs(actual_linear if linear_speed != 0 else actual_angular)
        duration = amount / max(actual_speed, 1e-6)

        self.movement = (actual_linear, actual_angular, time.monotonic(),
                         duration, description)

        self.get_logger().info(
            f'[MOVE START] {description} | '
            f'Amount={amount:.3f}{"m" if linear_speed != 0 else "rad"} | '
            f'Duration={duration:.2f}s | '
            f'v={actual_linear:+.3f}m/s | w={actual_angular:+.3f}rad/s'
        )

    def move(self):
        '''
        Burst of open-loop movement, then stop and wait for slam's pose,
        then plan the next burst from that pose.
        '''
        now = time.monotonic()

        # Stopped after a burst: wait until slam replies (see update_pose)
        if self.waiting_slam:
            assert self.movement is None
            self.command(0.0, 0.0)
            if now - self.last_wait_log > 2.0:
                self.last_wait_log = now
                self.get_logger().warning(f'Waiting for slam reply to burst {self.burst_count}')
            return

        # Startup wait / pause after reaching a waypoint
        if now < self.pause_until:
            self.command(0.0, 0.0)
            return

        # In a burst: keep driving, then stop and report it to slam
        if self.movement:
            linear, angular, start, duration, description = self.movement

            # Round to the nearest control tick instead of always overshooting
            if now - start < duration - CALLBACK_FREQUENCY / 2:
                self.command(linear, angular)
                return

            self.command(0.0, 0.0)
            dt = now - start                  # how long the motors were on
            start_window = self.ros_now()        # absolute time, must be ROS time

            self.burst_count += 1
            self.slam_publisher.publish(Float64MultiArray(data=[
                linear, angular, dt, start_window]))

            self.movement = None
            self.waiting_slam = True
            self.get_logger().info(
                f'[MOVE END] {description} | sent burst {self.burst_count}, dt={dt:.2f}s')
            return

        # All waypoints completed.
        if self.waypoint_index >= len(self.waypoints):
            self.command(0.0, 0.0)
            return

        # No pose from slam yet, or nothing new since the last decision: stay stopped.
        if self.pose is None or self.pose_time <= self.used_pose_time:
            self.command(0.0, 0.0)
            return
        self.used_pose_time = self.pose_time

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
                self.waypoint_index += 1
                self.position_reached = False
                self.pause_until = time.monotonic() + 1.0
                # No movement means no new slam pose, so mark the current
                # pose as new to let the planner run again after the pause.
                self.pose_time = time.monotonic()

                if self.waypoint_index >= len(self.waypoints):
                    self.get_logger().info('[DONE] All waypoints completed')
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