import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, TwistStamped
from std_msgs.msg import Float32MultiArray


# Existing motor calibration: [v, omega] = A @ [left, right].
CALIBRATION_MATRIX = np.array([
    [0.565034, 0.733764],
    [-4.08242, 3.82298],
])

MAX_MOTOR_COMMAND = 0.5
DRIVE_SPEED = 0.2                 # Requested forward speed (m/s)
TURN_SPEED = 3.0                  # Requested turning speed (rad/s)
POSITION_TOLERANCE = 0.05          # meters
HEADING_TOLERANCE = 0.2           # radians
MAX_DRIVE_SEGMENT = 0.20          # meters per movement burst
CALLBACK_FREQUENCY = 0.05         # 20 Hz command heartbeat
REPORT_PERIOD = 0.5              # seconds between trajectory samples
WAYPOINT_PAUSE = 1.0             # optional pause when a waypoint is reached

# EKF-SLAM defines the starting robot pose as (0, 0, 0).
OFFSET = (0.0, 0.0, 0.0)


def wrap_angle(angle):
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class MovementController(Node):
    def __init__(self):
        super().__init__('movement_controller_node')

        # The motor topic remains unchanged. SLAM additionally receives the
        # effective v/omega after motor commands have been limited to +/-0.5.
        self.motor_pub = self.create_publisher(
            Float32MultiArray, 'motor_commands', 10)
        self.velocity_pub = self.create_publisher(
            TwistStamped, '/slam/cmd_velocity', 10)

        # /slam/pose is the continuously predicted EKF estimate. It is used to
        # start navigation and to display the current position, not to decide
        # when a finished movement burst is ready for another one.
        self.create_subscription(Pose2D, '/slam/pose', self.update_pose, 10)

        # /slam/segment_pose is emitted once after each stop's observation
        # window. It arrives even if the camera never detects an AprilTag.
        self.create_subscription(
            Pose2D, '/slam/segment_pose', self.update_segment_pose, 10)

        # Preserve the existing waypoint format: x, y, theta per line.
        with open('./waypoint_3_square.txt') as file:
            self.waypoints = [
                (x + OFFSET[0], y + OFFSET[1], wrap_angle(theta + OFFSET[2]))
                for line in file if line.strip()
                for x, y, theta in [map(float, line.split(','))]
            ]

        # Waypoint 0 is the defined starting pose, so navigate toward #1.
        self.waypoint_index = 1
        self.pose = None                 # Latest continuous pose (or accepted segment pose)
        self.segment_pose = None         # Most recent finalized pose for navigation
        self.waiting_for_segment = False
        self.segment_pose_ready = False

        # An active burst stores (requested v, requested omega, effective v,
        # effective omega, monotonic start time, duration, description).
        self.movement = None
        self.pause_until = 0.0
        self.finished = False
        self.position_reached = False

        self.total_linear_distance_travelled = 0.0
        self.total_angular_distance_travelled = 0.0
        self.position_history = []

        self.create_timer(CALLBACK_FREQUENCY, self.move)
        self.create_timer(REPORT_PERIOD, self.report_position_history)

        #trajectory tracking
        self.trajectories = [[(0.0, 0.0, 0.0)]]
        self.position_history = []

        self.get_logger().info(
            f'[START] {len(self.waypoints)} waypoints | '
            f'drive={DRIVE_SPEED:.2f} m/s | turn={TURN_SPEED:.2f} rad/s')

    def update_pose(self, msg):
        """Record the live EKF estimate, including prediction-only updates."""
        self.pose = (msg.x, msg.y, msg.theta)
        self.trajectories[-1].append(self.pose)

    def update_segment_pose(self, msg):
        """Receive the final pose for a completed movement burst."""
        # A segment-ready message is only useful after our movement has stopped.
        # Ignores the original startup pose on the continuous topic (/pose).
        if not self.waiting_for_segment:
            return

        self.segment_pose = (msg.x, msg.y, msg.theta)
        self.segment_pose_ready = True

    def report_position_history(self):
        """Log the EKF trajectory; don't maintain a second dead-reckoning pose."""
        if self.pose is not None:
            self.position_history.append(self.pose)

    def command(self, linear_speed, angular_speed):
        """Send motor commands AND the corresponding effective velocities."""
        # Convert requested velocities into normalized motor inputs.
        left, right = np.linalg.solve(
            CALIBRATION_MATRIX, [linear_speed, angular_speed])

        # Scale both inputs equally so the motor command limits are respected.
        scale = max(1.0, abs(left) / MAX_MOTOR_COMMAND,
                    abs(right) / MAX_MOTOR_COMMAND)
        left, right = left / scale, right / scale

        # Convert the ACTUALLY SENT L/R values back to nominal physical speeds.
        # This avoids telling SLAM to integrate velocities that were saturated.
        effective_v, effective_omega = CALIBRATION_MATRIX @ [left, right]

        self.motor_pub.publish(Float32MultiArray(
            data=[float(left), float(right)]))

        velocity = TwistStamped()
        velocity.header.stamp = self.get_clock().now().to_msg()
        velocity.twist.linear.x = float(effective_v)
        velocity.twist.angular.z = float(effective_omega)
   
        self.velocity_pub.publish(velocity)

        return float(effective_v), float(effective_omega)

    def create_movement(self, linear_speed, angular_speed, amount, description):
        """Begin one timed burst using the speed after motor saturation."""
        # Clear any previous readiness BEFORE sending the new motor command.
        self.segment_pose_ready = False
        self.segment_pose = None

        effective_v, effective_omega = self.command(linear_speed, angular_speed)
        actual_speed = abs(effective_v if linear_speed != 0 else effective_omega)
        if actual_speed < 1e-6:
            self.get_logger().error('Requested movement produced zero effective speed')
            self.command(0.0, 0.0)
            return

        duration = amount / actual_speed
        self.movement = (linear_speed, angular_speed, effective_v,
                         effective_omega, time.monotonic(), duration, description)

        self.get_logger().info(
            f'[MOVE START] {description} | amount={amount:.3f} '
            f'{"m" if linear_speed != 0 else "rad"} | duration={duration:.2f}s | '
            f'v={effective_v:+.3f} m/s, w={effective_omega:+.3f} rad/s')

    def move(self):
        """Move, wait for a finalized EKF pose, then plan the next burst."""
        now = time.monotonic()

        # During a movement burst, repeat the command at 20 Hz.
        # At the scheduled end, STOP and wait for SLAM's segment pose.
        if self.movement is not None:
            v, omega, effective_v, effective_omega, start, duration, description = self.movement
            if now - start < duration:
                self.command(v, omega)
                return

            self.command(0.0, 0.0)
            self.total_linear_distance_travelled += abs(effective_v) * duration
            self.total_angular_distance_travelled += abs(effective_omega) * duration
            self.movement = None
            self.waiting_for_segment = True
            self.get_logger().info(
                f'[MOVE END] {description} | waiting for EKF segment pose')
            return

        # After stopping, keep sending zero heartbeats until SLAM completes
        # its settle + camera-observation window. This is NOT a tag search.
        if self.waiting_for_segment:
            self.command(0.0, 0.0)
            if not self.segment_pose_ready:
                return

            # The next waypoint decision uses this FINALIZED EKF estimate.
            # It is valid even when no AprilTag was visible during the pause.
            self.pose = self.segment_pose
            self.segment_pose_ready = False
            self.waiting_for_segment = False

        # Preserve the optional short pause on completed waypoints.
        if now < self.pause_until:
            self.command(0.0, 0.0)
            return

        # Wait for SLAM's initial pose instead of requiring a first tag.
        if self.pose is None:
            self.command(0.0, 0.0)
            return

        # Calculate the distance and direction to the current waypoint.
        x, y, theta = self.pose
        goal_x, goal_y, goal_theta = self.waypoints[self.waypoint_index]

        dx = goal_x - x
        dy = goal_y - y

        distance_remaining = math.hypot(dx, dy)

        # Heading needed to travel toward the waypoint.
        heading_error = wrap_angle(math.atan2(dy, dx) - theta)

        # Final heading requested by the waypoint file.
        final_heading_error = wrap_angle(goal_theta - theta)

        self.get_logger().info(
            f'[POSE] WP {self.waypoint_index} | '
            f'robot=({x:.3f}, {y:.3f}, {math.degrees(theta):+.1f}deg) | '
            f'goal=({goal_x:.3f}, {goal_y:.3f}, '
            f'{math.degrees(goal_theta):+.1f}deg)'
        )

        # Once we reach the waypoint's position, remember it.
        # This prevents a later EKF correction from making
        # the controller turn around and revisit the same corner.
        if distance_remaining <= POSITION_TOLERANCE:
            self.position_reached = True

        if self.position_reached:
            # Position has already been reached.
            # Only the final heading needs correction.
            if abs(final_heading_error) <= HEADING_TOLERANCE:
                self.get_logger().info(
                    f'[WAYPOINT REACHED] {self.waypoint_index}'
                )

                # Wrap around to continuously traverse the square.
                self.waypoint_index = (
                    self.waypoint_index + 1
                ) % len(self.waypoints)

                if self.waypoint_index == 0:
                    self.trajectories.append([self.pose])

                # Reset the position latch for the next waypoint.
                self.position_reached = False

            else:
                # Rotate in place to match the waypoint heading.
                self.create_movement(
                    0.0,
                    math.copysign(TURN_SPEED, final_heading_error),
                    abs(final_heading_error),
                    'FINAL TURN'
                )

        elif abs(heading_error) > HEADING_TOLERANCE:
            # Not at the waypoint, and not facing toward it.
            # Rotate before driving.
            self.create_movement(
                0.0,
                math.copysign(TURN_SPEED, heading_error),
                abs(heading_error),
                'TURN'
            )

        else:
            # Facing approximately toward the waypoint.
            # Drive a short distance, then stop for EKF estimation.
            self.create_movement(
                DRIVE_SPEED,
                0.0,
                min(distance_remaining, MAX_DRIVE_SEGMENT),
                'DRIVE'
            )

        self.get_logger().info(
            f'[WP {self.waypoint_index}] '
            f'position_reached={self.position_reached} '
            f'distance={distance_remaining:.3f} '
            f'heading_error={math.degrees(heading_error):+.1f}deg '
            f'final_heading_error={math.degrees(final_heading_error):+.1f}deg'
        )
        
    def save_trajectories(self):
        with open('robot_trajectories.txt', 'w') as file:
            for loop_number, trajectory in enumerate(self.trajectories, 1):
                if not trajectory:
                    continue

                file.write(f'LOOP {loop_number}\n')

                for x, y, theta in trajectory:
                    file.write(f'{x:.4f}, {y:.4f}, {theta:.4f}\n')

                file.write('\n')
        self.plot_trajectories()

    def plot_trajectories(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        trajectories = []
        current_loop = []

        with open("robot_trajectories.txt") as file:
            for line in file:
                line = line.strip()

                if line.startswith("LOOP"):
                    if current_loop:
                        trajectories.append(current_loop)
                    current_loop = []

                elif line:
                    x, y, theta = map(float, line.split(","))
                    current_loop.append((x, y))

        if current_loop:
            trajectories.append(current_loop)

        plt.figure(figsize=(8, 8))
        colors = mpl.colormaps["tab10"]

        # Plot each loop in a different color
        for i, trajectory in enumerate(trajectories):
            x, y = zip(*trajectory)

            color = colors(i % 10)
            plt.plot(x, y, color=color, label=f"Loop {i + 1}", linewidth=2)
            plt.scatter(x[0], y[0], color=color, marker="o", s=40)
            plt.scatter(x[-1], y[-1], color=color, marker="x", s=60)

        # Plot waypoints as large black dots
        for i, waypoint in enumerate(self.waypoints):
            wx, wy = waypoint[:2]

            plt.scatter(
                wx, wy,
                color="black",
                s=150,
                marker="o",
                edgecolors="white",
                linewidths=1.5,
                zorder=10,
                label="Waypoints" if i == 0 else None
            )

            # Label waypoint numbers
            plt.annotate(
                str(i + 1),
                (wx, wy),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold"
            )

        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.title("EKF-SLAM Robot Trajectories")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig("robot_trajectories.png", dpi=300)
        plt.close()



def main(args=None):
    rclpy.init(args=args)
    node = MovementController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.save_trajectories()
        finally:
            node.command(0.0, 0.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()

