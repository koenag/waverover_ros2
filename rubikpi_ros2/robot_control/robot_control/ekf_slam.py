"""slam.py: coordinator between movement_control and the EKF.

The EKF is a plain Python class (kalman_filter_code.py, class KalmanEKF) that
this node calls directly, so there are no ekf topics.

  /slam/cmd_velocity   (movement -> slam)   TwistStamped
      The effective [v, w] after motor saturation, repeated every control
      tick as a heartbeat (including zeros). Slam predicts with the current
      command all the time, so it also works with no tag in view.

  /slam/pose           (slam -> movement)   Pose2D, every TIMER_PERIOD
  /slam/segment_pose   (slam -> movement)   Pose2D, ONCE per stop, after the
      settle + observation window, whether or not a tag was seen.
  /slam/landmarks      (slam -> debugging)  [tag_id, x, y, ...]

After a burst the robot stops. Camera frames captured in the stationary
observation window [stop + SETTLE_TIME, stop + SETTLE_TIME + OBSERVATION_TIME]
are applied to the EKF, one image at a time.
"""

import numpy as np
import rclpy

from geometry_msgs.msg import Pose2D, TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage

from robot_control.kalman_filter_code import KalmanEKF

# Timing values are initial choices; verify how long the rover actually coasts
# and how long the camera detector takes to publish an image.
TIMER_PERIOD = 0.05      # publish the continuous pose at 20 Hz
MAX_PREDICT_STEP = 0.05  # keep individual motion integrations short
SETTLE_TIME = 0.3        # wait for physical motion to stop
OBSERVATION_TIME = 0.4   # accept stationary camera frames during this window
COMMAND_TIMEOUT = 0.3    # heartbeat limit while a nonzero command is active

ZERO_COMMAND = (0.0, 0.0)


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class RobotIter:
    """The observation window after one burst.

    Created when the robot stops and dropped as soon as it moves again.
    Camera frames are only used while one exists."""

    def __init__(self, stop_time):
        self.stop_time = stop_time
        self.valid_from = stop_time + SETTLE_TIME
        self.valid_until = self.valid_from + OBSERVATION_TIME
        self.seen_frames = set()  # capture times already applied
        self.published = False    # segment pose already sent


class SlamNode(Node):
    def __init__(self):
        super().__init__("slam_node")

        self.declare_parameter("camera_frame", "camera_frame")
        self.camera_frame = self.get_parameter("camera_frame").value

        # Camera origin in the robot frame and camera->robot rotation.
        self.camera_translation = np.array([0.095, 0.0, 0.127])
        roll = np.deg2rad(-80.0)
        yaw = np.deg2rad(-90.0)
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll),  np.cos(roll)],
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw),  np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.camera_rotation = Rz @ Rx

        self.ekf = KalmanEKF()
        self.ekf_time = None          # ROS time (s) the EKF is predicted up to
        self.command = ZERO_COMMAND   # current effective (v, w)
        self.last_command_time = None
        self.failed = False
        self.robot_iter = None
        self.last_state_log = 0.0

        self.create_subscription(
            TwistStamped, "/slam/cmd_velocity", self.velocity_callback, 10)
        self.create_subscription(TFMessage, "/tf", self.tf_callback, 100)
        self.create_timer(TIMER_PERIOD, self.timer_callback)

        self.pose_pub = self.create_publisher(Pose2D, "/slam/pose", 10)
        self.segment_pose_pub = self.create_publisher(Pose2D, "/slam/segment_pose", 10)
        self.landmark_pub = self.create_publisher(
            Float32MultiArray, "/slam/landmarks", 10)

        self.get_logger().info("slam_node ready")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def now_sec(self):
        """ROS time in seconds, or None if the clock isn't running or jumped back."""
        t = self.get_clock().now().nanoseconds * 1e-9
        if t <= 0:
            return None  # in simulation, /clock may not have started
        if self.ekf_time is not None and t < self.ekf_time:
            self.fail("ROS clock moved backwards")
            return None
        if self.ekf_time is None:
            self.ekf_time = t
        return t

    def fail(self, reason):
        """Stop publishing potentially inaccurate poses if motion is lost."""
        if not self.failed:
            self.failed = True
            self.get_logger().error(
                f"SLAM prediction is unreliable ({reason}); stop rover and restart SLAM.")

    def landmarks_flat(self):
        """[id1, mx1, my1, id2, ...] from the EKF state."""
        flat = []
        for tag_id, i in self.ekf.landmark_map.items():
            flat += [float(tag_id), float(self.ekf.state[i]),
                     float(self.ekf.state[i + 1])]
        return flat

    # ------------------------------------------------------------------
    # EKF
    # ------------------------------------------------------------------
    def invoke_ekf(self, v, w, dt, tags):
        """One EKF update: predict over dt, then apply the tags (either may be empty)."""
        try:
            self.ekf.update_state(v, w, dt, tags)
        except Exception as e:  # e.g. LinAlgError
            self.fail(f"ekf update failed: {e}")

    def predict_to(self, timestamp):
        """Advance the EKF to timestamp using the active command.

        Runs even when no tag has ever been seen. The real elapsed time is
        integrated (not the timer period), in short steps for accuracy."""
        if timestamp < self.ekf_time:
            self.fail("attempted to predict backward in time")
            return

        while self.ekf_time < timestamp and not self.failed:
            step = min(timestamp - self.ekf_time, MAX_PREDICT_STEP)
            self.invoke_ekf(*self.command, step, {})
            self.ekf_time = min(self.ekf_time + step, timestamp)

    # ------------------------------------------------------------------
    # Movement control -> slam: the command heartbeat
    # ------------------------------------------------------------------
    def velocity_callback(self, msg):
        """Update the command, integrating the OLD command before switching."""
        if self.failed:
            return
        now = self.now_sec()
        if now is None:
            return

        # Movement publishes the effective v, w AFTER motor saturation. The
        # receipt time approximates the moment the motors switched.
        new_command = (msg.twist.linear.x, msg.twist.angular.z)
        if not all(np.isfinite(new_command)):
            self.fail("invalid velocity command")
            return

        self.predict_to(now)
        if self.failed:
            return

        was_moving = self.command != ZERO_COMMAND
        is_moving = new_command != ZERO_COMMAND
        self.command = new_command
        self.last_command_time = now  # repeated commands are heartbeats

        if is_moving and not was_moving:
            self.robot_iter = None  # a new burst ends any earlier window
        elif was_moving and not is_moving:
            # ONE window per burst. Repeated zero commands don't restart it.
            self.robot_iter = RobotIter(now)

    # ------------------------------------------------------------------
    # Detector -> slam: use tags only inside the observation window
    # ------------------------------------------------------------------
    def tf_callback(self, msg):
        it = self.robot_iter
        if self.failed or it is None or it.published:
            return
        now = self.now_sec()
        if now is None or not it.valid_from <= now <= it.valid_until:
            return

        # /tf may carry several tags from the SAME image. Group by capture
        # stamp so each image is applied once, even if /tf republishes it.
        frames = {}
        for stamped in msg.transforms:
            parent = stamped.header.frame_id.lstrip("/")
            child = stamped.child_frame_id.lstrip("/")

            if parent != self.camera_frame:
                continue
            if not child.startswith("tag36h11:"):
                continue
            try:
                tag_id = int(child.split(":", 1)[1])
            except ValueError:
                continue

            # The frame must have been CAPTURED while stationary (processing
            # may lag, so arrival time alone isn't enough).
            capture_time = stamp_to_sec(stamped.header.stamp)
            if not it.valid_from <= capture_time <= now:
                continue
            if capture_time in it.seen_frames:
                continue

            t = stamped.transform.translation
            tag_in_camera = np.array([t.x, t.y, t.z])
            if not np.all(np.isfinite(tag_in_camera)) or tag_in_camera[2] <= 0:
                continue

            tag_in_robot = (
                self.camera_translation
                + self.camera_rotation @ tag_in_camera
            )
            if not np.all(np.isfinite(tag_in_robot)):
                continue

            frames.setdefault(capture_time, {})[tag_id] = tag_in_robot[:2]

        # The rover is stationary, so a valid image can be applied right away.
        for capture_time, tags in sorted(frames.items()):
            self.predict_to(now)
            if self.failed:
                return
            self.invoke_ekf(0.0, 0.0, 0.0, tags)
            it.seen_frames.add(capture_time)

    # ------------------------------------------------------------------
    # Timer: predict, publish the pose, and send the segment pose once
    # ------------------------------------------------------------------
    def timer_callback(self):
        if self.failed:
            return
        now = self.now_sec()
        if now is None:
            return

        # Missing heartbeats while moving mean the true motion is unknown;
        # don't silently assume the motors stopped.
        if (self.command != ZERO_COMMAND and self.last_command_time is not None
                and now - self.last_command_time > COMMAND_TIMEOUT):
            self.fail("missing velocity command heartbeat")
            return

        self.predict_to(now)
        if self.failed:
            return

        if now - self.last_state_log >= 1.0:
            self.get_logger().info(f"EKF state: {np.round(self.ekf.state, 3)}")
            self.last_state_log = now

        self.publish_pose(self.pose_pub)

        # Once per stop, after the observation window, even with zero tags.
        it = self.robot_iter
        if (self.command == ZERO_COMMAND and it is not None
                and not it.published and now >= it.valid_until):
            it.published = True
            self.publish_pose(self.segment_pose_pub)

        self.landmark_pub.publish(Float32MultiArray(data=self.landmarks_flat()))

    # ------------------------------------------------------------------
    # slam -> movement control
    # ------------------------------------------------------------------
    def publish_pose(self, publisher):
        """Publish the robot pose (x, y, theta); covariance stays in the EKF."""
        x, y, theta = self.ekf.state[:3]
        msg = Pose2D()
        msg.x, msg.y, msg.theta = float(x), float(y), float(theta)
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SlamNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()