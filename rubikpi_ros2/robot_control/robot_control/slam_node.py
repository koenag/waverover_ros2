from math import cos, isfinite, radians, sin
import re

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D, TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage

from .ekf_slam import EKFSLAM, MotionNoise, TagObservation


# Calibrate these values separately
MOTION_NOISE = MotionNoise(
    q_d_per_m=0.005,
    q_d_per_rad=0.001,
    q_theta_per_m=0.002,
    q_theta_per_rad=0.01,
)

# R IS a fixed 2x2 covariance for [tag_forward, tag_left], in meters squared.
# Replace with the matrix computed from stationary AprilTag measurements.
MEASUREMENT_R = np.array([
    [0.01, 0.0],
    [0.0, 0.01],
])

# Timing values are initial choices; verify how long the rover actually coasts
# and how long the camera detector takes to publish an image.
TIMER_PERIOD = 0.05      # Publish continuous pose at 20 Hz.
MAX_PREDICT_STEP = 0.05  # Keep individual motion integrations short.
SETTLE_TIME = 0.3       # Wait for physical motion to stop.
OBSERVATION_TIME = 0.4  # Accept stationary camera frames during this window.
COMMAND_TIMEOUT = 0.3  # Only check heartbeats while a nonzero command is active.

# The same calibration transform as the previous slam_node.py.
CAMERA_TRANSLATION = np.array([0.095, 0.0, 0.127])
CAMERA_RPY_DEG = (-80.0, 0.0, -90.0)
CAMERA_FRAME = 'camera_frame'
TAG_FRAME_PREFIX = 'tag36h11:'

NANOSECONDS = 1_000_000_000
ZERO_COMMAND = (0.0, 0.0)


class SLAMNode(Node):
    def __init__(self):
        super().__init__('ekf_slam_node')

        # One EKF contains the robot pose, mapped landmarks, and full covariance.
        # The EKF itself handles all mathematical prediction/correction steps.
        self.ekf = EKFSLAM(noise_config=MOTION_NOISE)

        # Current command and the time through which it has been integrated.
        # Use ROS time for both command receipt and camera capture timestamps.
        self.clock = self.get_clock()
        self.ekf_time = None
        self.command = ZERO_COMMAND
        self.last_command_time = None
        self.failed = False

        # Each moving -> stopped transition begins one observation cycle.
        # At the end we publish ONE segment pose, with or without any tags.
        self.stop_time = None
        self.segment_ready = False
        self.seen_frames = set()

        # Logging time tracking
        self.last_state_log = 0.0

        # Transform from camera optical coordinates to rover coordinates.
        # Robot axes: x forward, y left.  Keep the original calibrated values.
        roll, pitch, yaw = map(radians, CAMERA_RPY_DEG)
        rx = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)],
                       [0, sin(roll), cos(roll)]])
        ry = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0],
                       [-sin(pitch), 0, cos(pitch)]])
        rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0],
                       [0, 0, 1]])
        self.camera_rotation = rz @ ry @ rx
        self.tag_pattern = re.compile(re.escape(TAG_FRAME_PREFIX) + r'(\d+)')

        # Pose2D is sufficient for the movement controller (x, y, theta).
        # Float32MultiArray contains [tag_id, x, y] triples for debugging.
        self.pose_pub = self.create_publisher(Pose2D, '/slam/pose', 10)
        self.segment_pose_pub = self.create_publisher(Pose2D, '/slam/segment_pose', 10)
        self.landmark_pub = self.create_publisher(
            Float32MultiArray, '/slam/landmarks', 10)

        self.create_subscription(
            TwistStamped, '/slam/cmd_velocity', self.velocity_callback, 10)
        self.create_subscription(TFMessage, '/tf', self.camera_callback, 100)
        self.create_timer(TIMER_PERIOD, self.timer_callback)

    def now(self):
        """Get ROS time in nanoseconds, checking for a backwards clock jump."""
        t = self.clock.now().nanoseconds
        if t <= 0:
            return None  # In simulation, /clock may not have started.
        if self.ekf_time is not None and t < self.ekf_time:
            self.fail('ROS clock moved backwards')
            return None
        if self.ekf_time is None:
            self.ekf_time = t
        return t

    def fail(self, reason):
        """Stop publishing potentially inaccurate poses if motion is lost."""
        if not self.failed:
            self.failed = True
            self.get_logger().error(
                f'SLAM prediction is unreliable ({reason}); stop rover and restart SLAM.')

    def predict_to(self, timestamp):
        """Advance robot pose AND covariance using the active velocity command.

        This runs even when no AprilTag has ever been detected. Existing tag
        positions do not move during prediction, but the joint P is propagated.
        """
        if timestamp < self.ekf_time:
            self.fail('attempted to predict backward in time')
            return

        # Integrate the real elapsed time, not a hard-coded timer interval.
        # Splitting long intervals improves midpoint-integration accuracy.
        max_step_ns = round(MAX_PREDICT_STEP * NANOSECONDS)
        while self.ekf_time < timestamp:
            step = min(timestamp - self.ekf_time, max_step_ns)
            self.ekf.predict(*self.command, step / NANOSECONDS)
            self.ekf_time += step

    def velocity_callback(self, msg):
        """Update the command, integrating the OLD command before switching."""
        if self.failed:
            return
        now = self.now()
        if now is None:
            return

        # Controller must publish effective v and omega AFTER motor saturation.
        # Use callback receipt time as an approximate motor transition time.
        new_command = (msg.twist.linear.x, msg.twist.angular.z)
        if not all(isfinite(v) for v in new_command):
            self.fail('invalid velocity command')
            return

        self.predict_to(now)
        if self.failed:
            return

        was_moving = self.command != ZERO_COMMAND
        is_moving = new_command != ZERO_COMMAND
        self.command = new_command
        self.last_command_time = now  # Repeated commands are heartbeats.

        if is_moving and not was_moving:
            # A new motion burst begins; any earlier observation window ends.
            self.stop_time = None
            self.seen_frames.clear()
        elif was_moving and not is_moving:
            # Start ONE settle + observation window for this completed burst.
            # Repeated zero commands must not reset the start time.
            self.stop_time = now
            self.segment_ready = False
            self.seen_frames.clear()

    def camera_callback(self, msg):
        """Use only camera frames CAPTURED during a settled stationary period."""
        if self.failed:
            return
        now = self.now()
        if now is None or self.command != ZERO_COMMAND or self.stop_time is None:
            return
        if self.segment_ready:
            return  # The next segment-ready pose has already been published.

        # Camera processing may lag behind image capture. Both must fall in the
        # stationary interval; a frame captured while driving is not valid.
        valid_from = self.stop_time + round(SETTLE_TIME * NANOSECONDS)
        valid_until = valid_from + round(OBSERVATION_TIME * NANOSECONDS)
        if not valid_from <= now <= valid_until:
            return

        # /tf may contain multiple tags from the SAME image. Group by stamp,
        # and process each image only once (even if /tf republishes transforms).
        frames = {}
        for transform in msg.transforms:
            if transform.header.frame_id.lstrip('/') != CAMERA_FRAME:
                continue
            match = self.tag_pattern.fullmatch(transform.child_frame_id.lstrip('/'))
            if match is None:
                continue

            stamp = transform.header.stamp
            capture_time = stamp.sec * NANOSECONDS + stamp.nanosec
            if not valid_from <= capture_time <= now:
                continue
            if capture_time in self.seen_frames:
                continue

            p = transform.transform.translation
            p_camera = np.array([p.x, p.y, p.z])
            if not np.all(np.isfinite(p_camera)) or p_camera[2] <= 0:
                continue

            # Convert 3D camera coordinates to the planar robot measurement.
            p_robot = self.camera_rotation @ p_camera + CAMERA_TRANSLATION
            if not np.all(np.isfinite(p_robot)):
                continue

            tag_id = int(match.group(1))
            # EKF uses this R during measurement correction (not prediction).
            frames.setdefault(capture_time, {})[tag_id] = TagObservation(
                tag_id, p_robot[:2].copy(), MEASUREMENT_R.copy())

        # Since the rover stayed still, a valid image can be applied now;
        # no delayed-event buffer or EKF rollback is necessary.
        for capture_time, tags in sorted(frames.items()):
            self.predict_to(now)
            if self.failed:
                return
            self.ekf.process_observations(list(tags.values()))
            self.seen_frames.add(capture_time)

    def timer_callback(self):
        """Predict continuously and publish the segment pose once per stop."""
        if self.failed:
            return
        now = self.now()
        if now is None:
            return

        # Missing command heartbeats during movement means true motion is
        # unknown; do not silently assume the motor hardware stopped.
        if (self.command != ZERO_COMMAND and self.last_command_time is not None
                and now - self.last_command_time > COMMAND_TIMEOUT * NANOSECONDS):
            self.fail('missing velocity command heartbeat')
            return

        self.predict_to(now)
        if self.failed:
            return

        # Log the complete EKF state once per second.
        if now - self.last_state_log >= 1.0:
            self.get_logger().info(
                f"EKF state: {np.round(self.ekf.state, 3)}"
            )
            self.last_state_log = now

        # Publish a live estimate at every timer tick, with or without tags.
        self.publish_pose(self.pose_pub)

        # The pose is also published once as "ready" after the observation
        # window ends, even if zero tags were detected. No search is needed.
        if (self.command == ZERO_COMMAND and self.stop_time is not None
                and not self.segment_ready
                and now >= self.stop_time + (SETTLE_TIME + OBSERVATION_TIME) * NANOSECONDS):
            self.segment_ready = True
            self.publish_pose(self.segment_pose_pub)

        # Keep landmark output simple; it is just a flat list of ID/x/y triples.
        self.publish_landmarks()

    def publish_pose(self, publisher):
        """Publish the 2D robot pose; covariance stays inside EKFSLAM."""
        x, y, theta = self.ekf.get_robot_pose()
        msg = Pose2D()
        msg.x, msg.y, msg.theta = float(x), float(y), float(theta)
        publisher.publish(msg)

    def publish_landmarks(self):
        """Publish [tag_id, world_x, world_y, ...] for debugging/plotting."""
        data = []
        for tag_id, (position, _) in self.ekf.get_landmarks().items():
            data.extend([float(tag_id), float(position[0]), float(position[1])])
        self.landmark_pub.publish(Float32MultiArray(data=data))


def main(args=None):
    # A single-threaded executor avoids locks for this small node.
    rclpy.init(args=args)
    node = None
    try:
        node = SLAMNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
