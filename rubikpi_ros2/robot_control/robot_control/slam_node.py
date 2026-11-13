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

# Q values
MOTION_NOISE = MotionNoise(
    q_d_per_m=0.001,
    q_d_per_rad=0.0002,
    q_theta_per_m=0.002,
    q_theta_per_rad=0.004,
)

# R values
MEASUREMENT_R = np.array([
    [0.01, 0.0],
    [0.0, 0.01],
])

TIMER_PERIOD = 0.05      # Publish continuous pose at 20 Hz.
MAX_PREDICT_STEP = 0.05  # Keep individual motion integrations short.
SETTLE_TIME = 0.3       # Wait for physical motion to stop.
OBSERVATION_TIME = 0.5  # Accept stationary camera frames during this window.
COMMAND_TIMEOUT = 0.3  # Only check heartbeats while a nonzero command is active.

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
        self.ekf = EKFSLAM(noise_config=MOTION_NOISE)

        # Current command and the time through which it has been integrated.
        self.clock = self.get_clock()
        self.last_ekf_time = None
        self.command = ZERO_COMMAND
        self.last_command_time = None
        self.failed = False

        # Each moving -> stopped transition begins one observation cycle.
        # At the end we publish a segment pose, with or without any tags.
        self.robot_stop_time = None
        self.segment_ready = False
        self.seen_frames = set()

        # === State vector logging timer ===
        self.last_state_log = 0


        # Transform from camera optical coordinates to rover coordinates.
        # Robot axes: x forward, y left.
        roll, pitch, yaw = map(radians, CAMERA_RPY_DEG)
        rx = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)],
                       [0, sin(roll), cos(roll)]])
        ry = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0],
                       [-sin(pitch), 0, cos(pitch)]])
        rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0],
                       [0, 0, 1]])
        self.camera_rotation = rz @ ry @ rx
        self.tag_pattern = re.compile(re.escape(TAG_FRAME_PREFIX) + r'(\d+)')

        self.pose_pub = self.create_publisher(Pose2D, '/slam/pose', 10)
        self.segment_pose_pub = self.create_publisher(Pose2D, '/slam/segment_pose', 10)

        self.create_subscription(
            TwistStamped, '/slam/cmd_velocity', self.velocity_callback, 10)
        self.create_subscription(TFMessage, '/tf', self.camera_callback, 100)


        self.create_timer(TIMER_PERIOD, self.timer_callback)

    def now(self):
        t = self.get_clock().now().nanoseconds
        if self.last_ekf_time is None:
            self.last_ekf_time = t
        return t

    def fail(self, reason):
        """Stop publishing potentially inaccurate poses if motion is lost."""
        if not self.failed:
            self.failed = True
            self.get_logger().error(
                f'SLAM prediction is unreliable ({reason}); stop rover and restart SLAM.')

    def predict_to(self, timestamp):
        """Even when no april tag is detected run predict, but have the displacement count up to a specific timestep
        """
        if timestamp < self.last_ekf_time:
            self.fail('attempted to predict backward in time')
            return

        # Integrate the real elapsed time (with shorter intervals)
        max_step_ns = round(MAX_PREDICT_STEP * NANOSECONDS)
        while self.last_ekf_time < timestamp:
            step = min(timestamp - self.last_ekf_time, max_step_ns)
            self.ekf.predict(*self.command, step / NANOSECONDS)
            self.last_ekf_time += step

    def velocity_callback(self, msg):
        """Update the command, integrating the OLD command before switching."""
        if self.failed:
            return
        now = self.now()
        if now is None:
            return

        # Controller must publish effective v and omega.
        # Use callback receipt time as an approximate motor transition time.
        new_command = (msg.twist.linear.x, msg.twist.angular.z)
        if not all(isfinite(v) for v in new_command):
            self.fail('invalid velocity command')
            return

        self.predict_to(now)
        if self.failed:
            return

        # Publish a live estimate at every timer tick, with or without tags.
        self.publish_pose(self.pose_pub)

        was_moving = self.command != ZERO_COMMAND
        is_moving = new_command != ZERO_COMMAND

        movement_started = is_moving and not was_moving
        movement_stopped = was_moving and not is_moving

        self.command = new_command
        self.last_command_time = now

        if movement_started:
            self.robot_stop_time = None
            self.seen_frames.clear()
            self.get_logger().info(
                'SLAM | MOVING: estimating pose from velocity commands'
            )

        elif movement_stopped:
            self.robot_stop_time = now
            self.segment_ready = False
            self.seen_frames.clear()
            self.get_logger().info(
                'SLAM | STOPPED: waiting for camera observation window'
            )

    def camera_callback(self, msg):
        """Use only camera frames CAPTURED during a settled stationary period."""
        camera_observations_disabled = (
            self.failed
            or self.command != ZERO_COMMAND
            or self.robot_stop_time is None
            or self.segment_ready
        )

        if camera_observations_disabled:
            return

        now = self.now()
        if now is None:
            return
        if self.failed:
            return

        # Camera processing may lag behind image capture. Both must fall in the
        # stationary interval; a frame captured while driving is not valid.
        valid_from = self.robot_stop_time + round(SETTLE_TIME * NANOSECONDS)
        valid_until = valid_from + round(OBSERVATION_TIME * NANOSECONDS)
        
        observation_window_open = valid_from <= now <= valid_until
        if not observation_window_open:
            return

        # Process each tag once per capture time, ignoring duplicates and invalid frames.
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
            valid_camera_measurement = (
                np.all(np.isfinite(p_camera))
                and p_camera[2] > 0
            )

            if not valid_camera_measurement:
                continue

            # Convert 3D camera coordinates to the planar robot measurement.
            p_robot = self.camera_rotation @ p_camera + CAMERA_TRANSLATION
            if not np.all(np.isfinite(p_robot)):
                continue

            tag_id = int(match.group(1))
            # EKF uses this R during measurement correction (not prediction).
            frames.setdefault(capture_time, {})[tag_id] = TagObservation(
                tag_id, p_robot[:2].copy(), MEASUREMENT_R.copy())

        # Since the rover stayed still, a valid image can be applied now
        for capture_time, tags in sorted(frames.items()):
            self.predict_to(now)
            if self.failed:
                return

            # === Show tags passed to EKF and resulting pose correction ===
            # Estimates before EKF update
            pose_before = self.ekf.get_robot_pose()
            landmarks_before = {
                tag_id: position
                for tag_id, (position, _) in self.ekf.get_landmarks().items()
            }

            # Apply EKF updates
            results = self.ekf.process_observations(list(tags.values()))

            # Estimates after EKF update
            pose_after = self.ekf.get_robot_pose()
            landmarks_after = {
                tag_id: position
                for tag_id, (position, _) in self.ekf.get_landmarks().items()
            }

            # Robot pose correction
            delta = pose_after - pose_before
            delta[2] = self.ekf.wrap_angle(delta[2])

            self.get_logger().info(
                f'SLAM | ROBOT UPDATE: '
                f'dx={delta[0]:+.4f} m, '
                f'dy={delta[1]:+.4f} m, '
                f'dtheta={np.degrees(delta[2]):+.2f} deg'
            )

            # Landmark corrections
            for tag_id, position in landmarks_after.items():
                if tag_id not in landmarks_before:
                    self.get_logger().info(
                        f'SLAM | TAG {tag_id} INITIALIZED: '
                        f'({position[0]:.4f}, {position[1]:.4f})'
                    )
                else:
                    change = position - landmarks_before[tag_id]

                    self.get_logger().info(
                        f'SLAM | TAG {tag_id} UPDATE: '
                        f'dx={change[0]:+.4f} m, '
                        f'dy={change[1]:+.4f} m, '
                        f'distance={np.linalg.norm(change):.4f} m'
                    )
            self.seen_frames.add(capture_time)

    def timer_callback(self):
        """Predict continuously and publish the segment pose once per stop."""
        if self.failed:
            return
        now = self.now()
        if now is None:
            return

        # Missing command heartbeats during movement means true motion is unknown
        command_timed_out = (
            self.command != ZERO_COMMAND
            and self.last_command_time is not None
            and now - self.last_command_time > COMMAND_TIMEOUT * NANOSECONDS
        )

        if command_timed_out:
            self.fail('missing velocity command heartbeat')
            return

        self.predict_to(now)
        if self.failed:
            return

        # === Print complete EKF state once per second, even while stopped ===
        if now - self.last_state_log >= NANOSECONDS:
            tmp_x, tmp_y, tmp_theta = self.ekf.get_robot_pose()

            landmarks = [
                f'Tag {tag_id}: ({position[0]:.3f}, {position[1]:.3f})'
                for tag_id, (position, _) in self.ekf.get_landmarks().items()
            ]

            self.get_logger().info(
                f'SLAM | ROBOT: x={tmp_x:.3f}, y={tmp_y:.3f}, theta={tmp_theta:.3f} | '
                f'LANDMARKS: {", ".join(landmarks) if landmarks else "None"}'
            )
            self.last_state_log = now

        # Publish a live estimate at every timer tick, with or without tags.
        self.publish_pose(self.pose_pub)

        # The pose is also published once after the observation
        # window ends, even if zero tags were detected. No search for tags.
        segment_ready_to_publish = (
            self.command == ZERO_COMMAND
            and self.robot_stop_time is not None
            and not self.segment_ready
            and now >= self.robot_stop_time + (SETTLE_TIME + OBSERVATION_TIME) * NANOSECONDS
        )

        if segment_ready_to_publish:
            self.segment_ready = True
            self.publish_pose(self.segment_pose_pub)
            # === Final pose handed to the controller after this stop ===
            self.get_logger().info(
                f'SLAM | SEGMENT READY: pose={np.round(self.ekf.get_robot_pose(), 3)} '
                f'| camera_frames_used={len(self.seen_frames)}'
            )
        np.savetxt(
            "slam_final_covariance.txt",
            self.ekf.get_covariance_matrix(),
            fmt="%.8f"
        )


    def publish_pose(self, publisher):
        """Publish the 2D robot pose; covariance stays inside EKFSLAM."""
        x, y, theta = self.ekf.get_robot_pose()
        msg = Pose2D()
        msg.x, msg.y, msg.theta = float(x), float(y), float(theta)
        publisher.publish(msg)

    def log_final_state(self):
        state = self.ekf.get_state_vector()
        covariance = self.ekf.get_covariance_matrix()

        self.get_logger().info(
            f'\nSLAM | FINAL STATE:\n'
            f'{np.array2string(state, precision=5, suppress_small=True)}\n'
            f'SLAM | FINAL COVARIANCE MATRIX:\n'
            f'{np.array2string(covariance, precision=5, suppress_small=True, threshold=np.inf)}\n'
            f'SLAM | TAG ORDER: {list(self.ekf.landmark_indices.keys())}'
        
    )

    

def main(args=None):
    # A single-threaded executor avoids locks for this small node.
    rclpy.init(args=args)
    node = None
    try:
        node = SLAMNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        if node is not None:
            node.log_final_state()
        else:
            print("Node is None")
    finally:
        if node is not None:
            try:
                node.log_final_state()
            finally:
                node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
