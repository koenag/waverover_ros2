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
DRIVE_SPEED = 0.3
TURN_SPEED = 3.0
POSITION_TOLERANCE = 0.10
HEADING_TOLERANCE = 0.50
POSE_TIMEOUT = 3.0

SEARCH_TURN_SPEED = 3.0
SEARCH_INITIAL_PAUSE_TIME = 2.0
SEARCH_ROTATE_TIME = 0.10
SEARCH_PAUSE_TIME = 3.0

LOG_INTERVAL = 1.0
CONTROL_PERIOD = 0.05


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def parse_waypoints(filename):
    with open(filename) as file:
        return [tuple(map(float, line.split(',')))
                for line in file if line.strip()]


class WaypointController(Node):
    def __init__(self):
        super().__init__('waypoint_controller_node')
        self.publisher = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.create_subscription(Pose2D, '/robot_pose', self.pose_callback, 10)

        self.waypoints = parse_waypoints('./waypoint_2.txt')
        self.index = 1
        self.state = 'TURN'

        self.pose = None
        self.last_pose_time = 0.0
        self.used_pose_time = 0.0

        # Do not use AprilTag theta: start with the first waypoint's heading.
        self.estimated_theta = self.waypoints[0][2]
        self.last_heading_update = time.monotonic()

        # Current time-limited movement.
        self.move_end = None
        self.move_v = 0.0
        self.move_omega = 0.0

        self.search_phase = None
        self.search_phase_start = None

        self.last_log_time = 0.0
        self.last_reported_state = None
        self.motor_L = 0.0
        self.motor_R = 0.0
        self.last_v = 0.0
        self.last_omega = 0.0

        self.create_timer(CONTROL_PERIOD, self.update)
        self.get_logger().info(
            f'Started: {len(self.waypoints)} waypoints, '
            f'drive={DRIVE_SPEED:.2f}m/s, turn={TURN_SPEED:.2f}rad/s, '
            f'search={SEARCH_TURN_SPEED:.2f}rad/s, '
            f'position_tol={POSITION_TOLERANCE:.2f}m, '
            f'heading_tol={math.degrees(HEADING_TOLERANCE):.1f}deg, '
            f'heading source=command integration (AprilTag XY only)'
        )

    def pose_callback(self, msg):
        # Keep theta for logging only; it is never used for steering.
        self.pose = (msg.x, msg.y, msg.theta)
        self.last_pose_time = time.monotonic()

    def send_velocity(self, v, omega):
        L, R = CALIBRATION_INVERSE @ np.array([v, omega])
        scale = max(1.0, abs(L) / MAX_MOTOR_COMMAND,
                    abs(R) / MAX_MOTOR_COMMAND)
        self.motor_L, self.motor_R = float(L / scale), float(R / scale)
        self.last_v, self.last_omega = v / scale, omega / scale

        msg = Float32MultiArray()
        msg.data = [self.motor_L, self.motor_R]
        self.publisher.publish(msg)

    def update_heading(self, now):
        # Integrate the previously commanded angular speed before changing commands.
        dt = max(0.0, now - self.last_heading_update)
        self.estimated_theta = wrap_angle(
            self.estimated_theta + self.last_omega * dt
        )
        self.last_heading_update = now

    def start_move(self, v, omega, amount, now):
        self.send_velocity(v, omega)
        speed = abs(self.last_v if v != 0.0 else self.last_omega)
        duration = amount / max(speed, 1e-6)
        self.move_v = v
        self.move_omega = omega
        self.move_end = now + duration
        self.get_logger().info(
            f'{self.state}: amount={amount:.3f}, duration={duration:.2f}s'
        )

    def stop_move(self):
        self.move_end = None
        self.send_velocity(0.0, 0.0)

    def report(self, now, mode, distance=None, heading_error=None, final_error=None):
        if mode != self.last_reported_state or now - self.last_log_time >= LOG_INTERVAL:
            age = now - self.last_pose_time if self.pose is not None else float('inf')
            pose_text = ('none' if self.pose is None else
                         f'({self.pose[0]:.3f}, {self.pose[1]:.3f}, '
                         f'{math.degrees(self.pose[2]):.1f}deg)')
            goal_text = ('done' if self.index >= len(self.waypoints) else
                         f'{self.waypoints[self.index]}')
            errors = '' if distance is None else (
                f' dist={distance:.3f}m '
                f'heading_err={math.degrees(heading_error):+.1f}deg'
                f' final_err={math.degrees(final_error):+.1f}deg')
            self.get_logger().info(
                f'waypoint={self.index}/{len(self.waypoints)-1} state={mode} '
                f'goal={goal_text} pose={pose_text} age={age:.2f}s '
                f'est_theta={math.degrees(self.estimated_theta):+.1f}deg{errors} '
                f'cmd(v={self.last_v:.2f}, omega={self.last_omega:.2f}) '
                f'L={self.motor_L:+.3f} R={self.motor_R:+.3f}'
            )
            self.last_reported_state = mode
            self.last_log_time = now

    def update_search(self, now):
        if self.search_phase is None:
            self.search_phase = 'INITIAL_PAUSE'
            self.search_phase_start = now
            self.get_logger().warning('Localization unavailable; entering SEARCH')

        elapsed = now - self.search_phase_start
        if (self.search_phase == 'INITIAL_PAUSE' and
                elapsed >= SEARCH_INITIAL_PAUSE_TIME):
            self.search_phase = 'ROTATE'
            self.search_phase_start = now
        elif self.search_phase == 'ROTATE' and elapsed >= SEARCH_ROTATE_TIME:
            self.search_phase = 'PAUSE'
            self.search_phase_start = now
        elif self.search_phase == 'PAUSE' and elapsed >= SEARCH_PAUSE_TIME:
            self.search_phase = 'ROTATE'
            self.search_phase_start = now

        if self.search_phase == 'ROTATE':
            self.send_velocity(0.0, SEARCH_TURN_SPEED)
            mode = 'SEARCH_ROTATE'
        else:
            self.send_velocity(0.0, 0.0)
            mode = ('SEARCH_INITIAL_PAUSE' if self.search_phase == 'INITIAL_PAUSE'
                    else 'SEARCH_PAUSE')
        self.report(now, mode)

    def update(self):
        now = time.monotonic()
        self.update_heading(now)

        if self.index >= len(self.waypoints):
            self.stop_move()
            self.report(now, 'DONE')
            return

        new_pose = (self.pose is not None and
                    self.last_pose_time > self.used_pose_time)

        if new_pose:
            self.used_pose_time = self.last_pose_time
            self.stop_move()

            if self.search_phase is not None:
                self.get_logger().info('Localization reacquired; replanning XY')
                self.search_phase = None
                self.search_phase_start = None

            x, y, _ = self.pose  # Ignore AprilTag orientation for control.
            goal_x, goal_y, goal_theta = self.waypoints[self.index]
            dx, dy = goal_x - x, goal_y - y
            distance = math.hypot(dx, dy)
            heading_error = wrap_angle(math.atan2(dy, dx) - self.estimated_theta)
            final_error = wrap_angle(goal_theta - self.estimated_theta)

            if distance <= POSITION_TOLERANCE:
                if abs(final_error) <= HEADING_TOLERANCE:
                    self.get_logger().info(f'WAYPOINT {self.index} REACHED')
                    self.index += 1
                    self.state = 'TURN'
                else:
                    self.state = 'FINAL_TURN'
                    self.start_move(0.0, math.copysign(TURN_SPEED, final_error),
                                    abs(final_error), now)
            elif abs(heading_error) > HEADING_TOLERANCE:
                self.state = 'TURN'
                self.start_move(0.0, math.copysign(TURN_SPEED, heading_error),
                                abs(heading_error), now)
            else:
                self.state = 'DRIVE'
                self.start_move(DRIVE_SPEED, 0.0, distance, now)

            self.report(now, self.state, distance, heading_error, final_error)
            return

        if self.move_end is not None:
            if now < self.move_end:
                self.send_velocity(self.move_v, self.move_omega)
                self.report(now, self.state)
                return
            self.stop_move()
            self.state = 'WAIT_FOR_POSE'

        self.send_velocity(0.0, 0.0)
        if self.pose is None or now - self.last_pose_time > POSE_TIMEOUT:
            self.update_search(now)
        else:
            self.report(now, 'WAIT_FOR_POSE')

    def destroy_node(self):
        self.update_heading(time.monotonic())
        self.stop_move()
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
