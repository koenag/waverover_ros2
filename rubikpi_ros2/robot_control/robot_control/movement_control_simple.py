#!/usr/bin/env python3
"""ROS2 waypoint follower with continuous feedback and AprilTag-assisted odometry.

Topics:
  subscribe: /robot_pose      geometry_msgs/Pose2D  (world-frame x,y,theta)
  publish:   motor_commands   std_msgs/Float32MultiArray [L, R]

Assumptions:
  - Waypoints and /robot_pose use the SAME world coordinate frame.
  - The robot starts at the FIRST waypoint, including its initial heading.
  - Calibration maps [left_command, right_command] to [v_mps, omega_rad_s].
  - Positive omega is counterclockwise, with +X at theta=0.
"""
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


CALIBRATION_MATRIX = np.array([
    [0.565034, 0.733764],
    [-4.08242, 3.82298],
], dtype=float)
CALIBRATION_INVERSE = np.linalg.inv(CALIBRATION_MATRIX)

WAYPOINT_FILE = "./waypoint_2.txt"
CONTROL_PERIOD = 0.05
LOG_INTERVAL = 1.0
MAX_MOTOR_COMMAND = 0.5

# Closed-loop controller tuning
MAX_FORWARD_SPEED = 0.26        # m/s; modest speed for accuracy
MIN_FORWARD_SPEED = 0.13        # avoid motor dead zone
MAX_TURN_SPEED = 1.6            # rad/s
MIN_TURN_SPEED = 0.85           # in-place turn minimum
DRIVE_HEADING_GAIN = 2.8
TURN_HEADING_GAIN = 2.5
TURN_IN_PLACE_THRESHOLD = math.radians(25.0)
POSITION_TOLERANCE = 0.12       # meters
FINAL_HEADING_TOLERANCE = math.radians(10.0)
WAYPOINT_CONFIRM_TICKS = 3

# Localization fusion: conservative corrections to reduce AprilTag jitter.
POSE_XY_GAIN = 0.55
POSE_THETA_GAIN = 0.30
MAX_POSITION_INNOVATION = 0.75  # reject implausible position jumps
MAX_HEADING_INNOVATION = math.radians(65.0)
POSE_STALE_AFTER = 2.0

# Safety: odometry without sensors drifts. Stop rather than drive indefinitely
# with no new AprilTag observations. Increase cautiously if necessary.
MAX_TIME_WITHOUT_TAG = 30.0     # seconds after last accepted tag
MAX_DISTANCE_WITHOUT_TAG = 4.0  # meters of predicted travel
MAX_CONTROL_GAP = 0.5          # seconds, timer-stall failsafe


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value, low, high):
    return max(low, min(high, value))


def load_waypoints(filename):
    result = []
    with open(filename, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            values = [float(v.strip()) for v in line.split(",")]
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"Invalid waypoint on line {line_no}: {line}")
            result.append((values[0], values[1], wrap_angle(values[2])))
    if len(result) < 2:
        raise ValueError("Waypoint file must contain at least two x,y,theta rows")
    return result


class WaypointController(Node):
    def __init__(self):
        super().__init__("waypoint_controller_node")
        self.publisher = self.create_publisher(
            Float32MultiArray, "motor_commands", 10
        )
        self.subscription = self.create_subscription(
            Pose2D, "/robot_pose", self.pose_callback, 10
        )

        self.waypoints = load_waypoints(WAYPOINT_FILE)
        self.index = 1
        self.x, self.y, self.theta = self.waypoints[0]
        self.last_update = time.monotonic()
        self.start_time = self.last_update
        self.last_tag_time = None
        self.blind_distance = 0.0
        self.pending_pose = None
        self.tag_seen = False

        self.cmd_l = 0.0
        self.cmd_r = 0.0
        self.model_v = 0.0
        self.model_omega = 0.0
        self.confirm_ticks = 0
        self.mode = "STARTUP"
        self.last_log = 0.0
        self.finished = False
        self.safety_stopped = False

        self.timer = self.create_timer(CONTROL_PERIOD, self.update)
        self.get_logger().info(
            f"Closed-loop controller: {len(self.waypoints)} waypoints; "
            f"start={self.waypoints[0]}; target={self.waypoints[1]}; "
            "continuous odometry, AprilTag fusion, no search spinning."
        )

    def pose_callback(self, msg):
        values = (float(msg.x), float(msg.y), float(msg.theta))
        if all(math.isfinite(v) for v in values):
            self.pending_pose = (values[0], values[1], wrap_angle(values[2]),
                                 time.monotonic())

    def publish_command(self, v, omega):
        """Convert desired physical velocities to bounded motor commands."""
        l, r = CALIBRATION_INVERSE @ np.array([v, omega], dtype=float)
        scale = max(1.0, abs(l) / MAX_MOTOR_COMMAND,
                    abs(r) / MAX_MOTOR_COMMAND)
        l, r = float(l / scale), float(r / scale)
        # Model-predicted achieved velocity, including saturation.
        self.model_v, self.model_omega = (
            CALIBRATION_MATRIX @ np.array([l, r], dtype=float)
        )
        self.cmd_l, self.cmd_r = l, r
        msg = Float32MultiArray()
        msg.data = [l, r]
        self.publisher.publish(msg)

    def integrate_odometry(self, dt):
        """Integrate the PREVIOUS motor command before publishing a new one."""
        v, omega = self.model_v, self.model_omega
        theta0 = self.theta
        if abs(omega) < 1e-8:
            self.x += v * math.cos(theta0) * dt
            self.y += v * math.sin(theta0) * dt
        else:
            theta1 = theta0 + omega * dt
            self.x += v / omega * (math.sin(theta1) - math.sin(theta0))
            self.y += v / omega * (math.cos(theta0) - math.cos(theta1))
        self.theta = wrap_angle(theta0 + omega * dt)
        self.blind_distance += abs(v) * dt

    def fuse_tag(self, now):
        """Fuse the newest AprilTag observation without blindly trusting jumps."""
        if self.pending_pose is None:
            return
        mx, my, mt, received = self.pending_pose
        self.pending_pose = None

        if now - received > POSE_STALE_AFTER:
            return

        position_error = math.hypot(mx - self.x, my - self.y)
        angle_error = wrap_angle(mt - self.theta)

        # The first reading may have a moderate initial-frame discrepancy.
        # Still reject very large jumps: waypoint and tag frames must match.
        xy_limit = 1.50 if not self.tag_seen else MAX_POSITION_INNOVATION
        angle_limit = math.radians(90) if not self.tag_seen else MAX_HEADING_INNOVATION
        if position_error > xy_limit or abs(angle_error) > angle_limit:
            self.get_logger().warning(
                f"Rejected AprilTag jump: xy={position_error:.2f}m, "
                f"theta={math.degrees(angle_error):+.1f}deg"
            )
            return

        if not self.tag_seen:
            # First accepted tag establishes absolute world-frame pose.
            self.x, self.y, self.theta = mx, my, mt
            self.tag_seen = True
            self.get_logger().info("First AprilTag pose accepted")
        else:
            self.x += POSE_XY_GAIN * (mx - self.x)
            self.y += POSE_XY_GAIN * (my - self.y)
            self.theta = wrap_angle(
                self.theta + POSE_THETA_GAIN * angle_error
            )
        self.last_tag_time = now
        self.blind_distance = 0.0

    def safe_to_continue(self, now):
        # Allow starting without a tag, provided the first waypoint really
        # describes the initial pose. The same blind-travel limit applies.
        elapsed = now - (self.last_tag_time
                         if self.last_tag_time is not None
                         else self.start_time)
        return (elapsed <= MAX_TIME_WITHOUT_TAG and
                self.blind_distance <= MAX_DISTANCE_WITHOUT_TAG)

    def compute_control(self):
        gx, gy, gt = self.waypoints[self.index]
        dx, dy = gx - self.x, gy - self.y
        distance = math.hypot(dx, dy)
        final_error = wrap_angle(gt - self.theta)

        if distance <= POSITION_TOLERANCE:
            if abs(final_error) <= FINAL_HEADING_TOLERANCE:
                self.confirm_ticks += 1
                if self.confirm_ticks >= WAYPOINT_CONFIRM_TICKS:
                    self.get_logger().info(
                        f"Reached waypoint {self.index}/{len(self.waypoints)-1}: "
                        f"est=({self.x:.2f}, {self.y:.2f}, "
                        f"{math.degrees(self.theta):.1f}deg)"
                    )
                    self.index += 1
                    self.confirm_ticks = 0
                    if self.index >= len(self.waypoints):
                        self.finished = True
                        return 0.0, 0.0, "DONE"
                    # Stop for one control cycle at the waypoint.
                    return 0.0, 0.0, "NEXT_WAYPOINT"
                return 0.0, 0.0, "VERIFY_WAYPOINT"

            self.confirm_ticks = 0
            omega = clamp(TURN_HEADING_GAIN * final_error,
                          -MAX_TURN_SPEED, MAX_TURN_SPEED)
            if abs(omega) < MIN_TURN_SPEED:
                omega = math.copysign(MIN_TURN_SPEED, final_error)
            return 0.0, omega, "FINAL_TURN"

        self.confirm_ticks = 0
        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_heading - self.theta)

        if abs(heading_error) > TURN_IN_PLACE_THRESHOLD:
            omega = clamp(TURN_HEADING_GAIN * heading_error,
                          -MAX_TURN_SPEED, MAX_TURN_SPEED)
            if abs(omega) < MIN_TURN_SPEED:
                omega = math.copysign(MIN_TURN_SPEED, heading_error)
            return 0.0, omega, "ALIGN"

        # Continuous feedback while driving: steer on every timer tick.
        v = clamp(0.9 * distance, MIN_FORWARD_SPEED, MAX_FORWARD_SPEED)
        v *= max(0.55, math.cos(heading_error))
        omega = clamp(DRIVE_HEADING_GAIN * heading_error,
                      -MAX_TURN_SPEED, MAX_TURN_SPEED)
        return v, omega, "DRIVE"

    def update(self):
        now = time.monotonic()
        dt = now - self.last_update
        self.last_update = now

        if dt > MAX_CONTROL_GAP:
            self.publish_command(0.0, 0.0)
            self.safety_stopped = True
            self.get_logger().error(
                f"Controller stalled for {dt:.2f}s; stopping for safety"
            )
        if self.safety_stopped or self.finished:
            self.publish_command(0.0, 0.0)
            return

        # Predict first using the previous command, then correct with sensor.
        self.integrate_odometry(max(0.0, dt))
        self.fuse_tag(now)

        if not self.safe_to_continue(now):
            self.safety_stopped = True
            self.publish_command(0.0, 0.0)
            self.get_logger().error(
                "Dead-reckoning safety limit exceeded; motors stopped. "
                "Reacquire AprilTag or restart after verifying pose."
            )
            return

        v, omega, self.mode = self.compute_control()
        self.publish_command(v, omega)

        if now - self.last_log >= LOG_INTERVAL or self.mode in ("DONE", "NEXT_WAYPOINT"):
            age = (now - self.last_tag_time if self.last_tag_time is not None
                   else float("inf"))
            goal = (self.waypoints[self.index] if not self.finished else "done")
            self.get_logger().info(
                f"wp={self.index}/{len(self.waypoints)-1} "
                f"mode={self.mode} goal={goal} "
                f"est=({self.x:.3f},{self.y:.3f},"
                f"{math.degrees(self.theta):+.1f}deg) "
                f"tag_age={age:.1f}s blind={self.blind_distance:.2f}m "
                f"cmd(v={self.model_v:+.2f},w={self.model_omega:+.2f}) "
                f"L={self.cmd_l:+.3f} R={self.cmd_r:+.3f}"
            )
            self.last_log = now

    def destroy_node(self):
        self.publish_command(0.0, 0.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WaypointController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
