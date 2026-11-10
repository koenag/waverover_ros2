"""slam.py: coordinator between movement_control and the EKF.

The EKF is a plain Python class (kalman_filter_code.py, class KalmanEKF) that
this node calls directly, so there are no ekf topics.

  robot_burst_start    (movement -> slam)
      [v, w, start_time, duration]
      Sent when a burst STARTS. Slam uses it only to show a live pose while
      the robot is moving. Nothing is committed to the EKF from it.

  robot_movement_vw    (movement -> slam)
      [v, w, dt, start_window]
      Sent when a burst ENDS. Absolute ROS-clock times in seconds. Tags whose
      image stamp is in [start_window + SETTLE_TIME,
      start_window + SETTLE_TIME + DELAY] are accepted.

  /slam/pose           (slam -> movement)   Pose2D, every POSE_PUBLISH_PERIOD
      The best current pose. While a burst is running it is the committed
      pose extrapolated along the burst. Movement uses it for the starting
      pose and for monitoring only, never as a reply.

  verified_pose_update (slam -> movement)
      [ok, x, y, theta, n_landmarks, id1, mx1, my1, ...]
      Exactly once after each commit. Movement waits for it.
      ok = 1.0 normal, 0.0 if an EKF step failed (pose is then the
      last good pose)

When slam accepts tags (one state at a time):
  IDLE    no burst known.                      Tags ignored.
  MOVING  robot_burst_start received.          Tags ignored.
  WINDOW  robot_movement_vw received, before   Tags accepted if their image
          the commit timer fires.              stamp is inside the window.
  A commit (or an abandoned window) returns to IDLE.

Edge cases:
  - End message with no start message: works (WINDOW directly).
  - Start message while a window is still pending: the old window is
    abandoned (its commit timer is cancelled, no reply is sent for it).
  - End message never arrives: the live pose stops extrapolating at
    duration + END_MESSAGE_GRACE and falls back to the committed pose.
"""

import numpy as np
import rclpy

from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

from robot_control.kalman_filter_code import KalmanEKF

DELAY = 0.5                # length of the camera capture window
SETTLE_TIME = 0.3          # wait for the rover to physically stop
# Extra wait after the window closes so frames captured near its end (which
# reach slam late) still count. Set it to a bit more than the max "latency"
# printed in the commit log. The reply comes SETTLE + DELAY + this after the stop.
DETECTOR_LATENCY = 0.3
POSE_PUBLISH_PERIOD = 0.2  # how often /slam/pose is published
END_MESSAGE_GRACE = 1.0    # stop extrapolating this long after the planned end


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class LiveBurst:
    """A burst that has started but whose end message hasn't arrived yet."""

    def __init__(self, v, w, start_time, duration):
        self.v = v
        self.w = w
        self.start_time = start_time
        self.duration = duration
        self.expired = False       # True once the end message is overdue


class RobotIter:
    """One stop-and-go iteration.

    Created when movement_control publishes robot_movement_vw and set to
    None once the pose has been published back. Frames are only applied
    while one exists."""

    def __init__(self, v, w, dt, start_window):
        self.v = v
        self.w = w
        self.dt = dt
        self.start_window = start_window
        self.ok = True             # False if any EKF step failed
        self.seen_stamps = set()   # image stamps already applied
        self.frames_used = 0
        self.tag_ids = set()
        self.max_latency = 0.0     # worst (arrival - image stamp), seconds


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
        self.robot_iter = None
        self.live_burst = None
        self.commit_timer = None

        self.create_subscription(
            Float64MultiArray, "robot_burst_start", self.burst_start_callback, 10)
        self.create_subscription(
            Float64MultiArray, "robot_movement_vw", self.segment_callback, 10)
        self.create_subscription(
            TFMessage, "/tf", self.tf_callback, 100)

        self.pose_pub = self.create_publisher(
            Float64MultiArray, "verified_pose_update", 10)
        self.live_pose_pub = self.create_publisher(Pose2D, "/slam/pose", 10)

        # Periodic pose. The first message is also how movement learns the
        # starting pose, so there is no one-shot startup publish any more.
        self.create_timer(POSE_PUBLISH_PERIOD, self.publish_live_pose)

        self.get_logger().info("slam_node ready")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def cancel_timer(self, timer):
        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)
        return None

    def landmarks_flat(self):
        """[id1, mx1, my1, id2, ...] from the EKF state."""
        flat = []
        for tag_id, i in self.ekf.landmark_map.items():
            flat += [float(tag_id), float(self.ekf.state[i]),
                     float(self.ekf.state[i + 1])]
        return flat

    def safe_ekf_step(self, it, v, w, dt, tags, what):
        """Run one EKF step; on failure restore the previous state."""
        state, P = self.ekf.state.copy(), self.ekf.P.copy()
        landmark_map = dict(self.ekf.landmark_map)
        try:
            self.ekf.update_state(v, w, dt, tags)
        except Exception as e:  # e.g. LinAlgError; never leave movement hanging
            self.ekf.state, self.ekf.P = state, P
            self.ekf.landmark_map = landmark_map
            it.ok = False
            self.get_logger().error(f"ekf {what} failed: {e}")

    # ------------------------------------------------------------------
    # Movement control -> slam: a burst starts (MOVING)
    # ------------------------------------------------------------------
    def burst_start_callback(self, msg):
        d = msg.data
        if len(d) < 4 or not np.all(np.isfinite(d[:4])) or d[3] < 0:
            self.get_logger().error(
                "robot_burst_start needs 4 finite values [v, w, start, duration]")
            return

        # A new burst while the previous window is still open: movement moved
        # on without waiting, so the old reply would be stale. Abandon it.
        if self.robot_iter is not None:
            self.get_logger().warning(
                "burst started while the previous window was pending; abandoning it")
            self.commit_timer = self.cancel_timer(self.commit_timer)
            self.robot_iter = None

        self.live_burst = LiveBurst(
            v=float(d[0]), w=float(d[1]),
            start_time=float(d[2]), duration=float(d[3]))

    # ------------------------------------------------------------------
    # Movement control -> slam: the burst ended; predict (WINDOW)
    # ------------------------------------------------------------------
    def segment_callback(self, msg):
        d = msg.data
        if len(d) < 4:
            self.get_logger().error("robot_movement_vw needs 4 values")
            return

        # The burst is over, whether or not we saw its start message.
        self.live_burst = None

        it = RobotIter(
            v=float(d[0]), w=float(d[1]), dt=float(d[2]),
            start_window=float(d[3]),
        )
        self.robot_iter = it

        # Predict NOW (robot is already stopped), so every frame that arrives
        # afterwards corrects the post-motion pose. Empty dict = predict only.
        self.safe_ekf_step(it, it.v, it.w, it.dt, {}, "predict")

        # Publish after the capture window plus detector latency.
        self.commit_timer = self.cancel_timer(self.commit_timer)
        wait = max(0.001, it.start_window + SETTLE_TIME + DELAY
                   + DETECTOR_LATENCY - self.now_sec())
        self.commit_timer = self.create_timer(wait, self.commit)

    # ------------------------------------------------------------------
    # Detector -> slam: correct with each frame as it arrives
    # ------------------------------------------------------------------
    def tf_callback(self, msg):
        # Only in WINDOW: not while a burst is running, not when idle.
        if self.live_burst is not None:
            return
        it = self.robot_iter
        if it is None:
            return

        window_start = it.start_window + SETTLE_TIME
        window_end = window_start + DELAY
        now = self.now_sec()

        # One /tf message can hold several tags from the same image.
        # Group them by image stamp: one frame = one EKF correction.
        frames = {}
        for stamped in msg.transforms:
            parent = stamped.header.frame_id.lstrip("/")
            child = stamped.child_frame_id.lstrip("/")

            if parent != self.camera_frame:
                continue
            if not child.startswith("tag36h11:"):
                continue

            # Only images captured while settled and stationary.
            t_img = stamp_to_sec(stamped.header.stamp)
            if not (window_start <= t_img <= window_end):
                continue
            if t_img in it.seen_stamps:
                continue  # this image was already applied

            try:
                tag_id = int(child.split(":", 1)[1])
            except ValueError:
                continue

            t = stamped.transform.translation
            tag_in_camera = np.array([t.x, t.y, t.z])
            if not np.all(np.isfinite(tag_in_camera)) or tag_in_camera[2] <= 0:
                continue  # tag must be in front of the camera

            tag_in_robot = (
                self.camera_translation
                + self.camera_rotation @ tag_in_camera
            )
            frames.setdefault(t_img, {})[tag_id] = tag_in_robot[:2].copy()

        # Apply each frame immediately, oldest first. dt = 0 -> correction only.
        for t_img in sorted(frames):
            tags = frames[t_img]
            pose_before = self.ekf.state[:3].copy()
            self.safe_ekf_step(it, 0.0, 0.0, 0.0, tags, "correction")
            pose_after = self.ekf.state[:3]
            it.seen_stamps.add(t_img)
            it.frames_used += 1
            it.tag_ids.update(tags)

            latency = now - t_img
            it.max_latency = max(it.max_latency, latency)
            self.get_logger().info(
                f"frame t={t_img - it.start_window:+.2f}s after stop, "
                f"latency={latency:.2f}s, tags={sorted(tags)} | pose "
                f"{np.round(pose_before, 3)} -> {np.round(pose_after, 3)}")

    # ------------------------------------------------------------------
    # End of window: hand the pose back to movement
    # ------------------------------------------------------------------
    def commit(self):
        self.commit_timer = self.cancel_timer(self.commit_timer)
        it = self.robot_iter
        if it is None:
            return

        self.robot_iter = None  # tags are ignored until the next burst
        x, y, theta = self.ekf.state[:3]
        self.get_logger().info(
            f"commit: frames={it.frames_used} tags={sorted(it.tag_ids)} "
            f"max_latency={it.max_latency:.2f}s "
            f"pose=({x:.3f}, {y:.3f}, {np.degrees(theta):+.1f}deg)")
        self.get_logger().info(
            f"Robot pose: {np.round(self.ekf.state[:3], 4)}"
        )

        for tag_id, i in self.ekf.landmark_map.items():
            self.get_logger().info(
                f"Tag {tag_id}: {np.round(self.ekf.state[i:i+2], 4)}"
            )
        self.publish_pose(1.0 if it.ok else 0.0)

    # ------------------------------------------------------------------
    # slam -> movement control
    # ------------------------------------------------------------------
    def live_state(self):
        """Pose state now: the EKF state, extrapolated along a running burst.

        Works on a copy, so the EKF itself is only changed by the end message
        and the camera corrections."""
        state = self.ekf.state.copy()
        lb = self.live_burst
        if lb is None:
            return state

        elapsed = self.now_sec() - lb.start_time
        if elapsed > lb.duration + END_MESSAGE_GRACE:
            if not lb.expired:
                lb.expired = True
                self.get_logger().warning(
                    "burst end message is overdue; live pose is no longer extrapolated")
            return state

        elapsed = min(max(elapsed, 0.0), lb.duration)
        state, _, _ = self.ekf.pred_state(state, lb.v, lb.w, elapsed)
        return state

    def publish_live_pose(self):
        x, y, theta = self.live_state()[:3]
        msg = Pose2D()
        msg.x, msg.y, msg.theta = float(x), float(y), float(theta)
        self.live_pose_pub.publish(msg)

    def publish_pose(self, ok):
        """Send the robot pose and landmarks to movement control."""
        pose = self.ekf.state[:3]
        out = Float64MultiArray()
        out.data = (
            [float(ok), float(pose[0]), float(pose[1]),
             float(pose[2]), float(len(self.ekf.landmark_map))]
            + self.landmarks_flat()
        )
        self.pose_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()