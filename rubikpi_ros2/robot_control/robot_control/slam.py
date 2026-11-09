"""slam.py: coordinator between movement_control and the EKF.

The EKF is a plain Python class (kalman_filter_code.py, class KalmanEKF) that
this node calls directly, so there are no ekf topics.

  robot_movement_vw    (movement -> slam)
      [v, w, dt, start_window]
      Absolute ROS-clock times in seconds. Tags whose image stamp is in
      [start_window + SETTLE_TIME, start_window + SETTLE_TIME + DELAY]
      are accepted.

  verified_pose_update (slam -> movement)
      [ok, x, y, theta, n_landmarks, id1, mx1, my1, ...]
      Published once at startup (the starting pose), then exactly once
      after each commit. Movement waits for it, so no sequence number.
      ok = 1.0 normal, 0.0 if an EKF step failed (pose is then the
      last good pose)

Flow per iteration:
  1. robot_movement_vw arrives -> predict ONCE with (v, w, dt).
  2. Each camera frame in the window -> one correction step, applied
     immediately as it arrives (one update per frame, not one per stop).
  3. At window end + detector latency -> publish the pose.
"""

import numpy as np
import rclpy

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

from robot_control.kalman_filter_code import KalmanEKF

STARTUP_POSE_DELAY = 2.0  # seconds after launch before the starting pose is sent
DELAY = 0.4               # length of the camera capture window
SETTLE_TIME = 0.3         # wait for the rover to physically stop
DETECTOR_LATENCY = 0.0    # extra time for late /tf messages from the window

def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


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
        self.commit_timer = None

        self.create_subscription(
            Float64MultiArray, "robot_movement_vw", self.segment_callback, 10)
        self.create_subscription(
            TFMessage, "/tf", self.tf_callback, 100)

        self.pose_pub = self.create_publisher(
            Float64MultiArray, "verified_pose_update", 10)

        # Publish the starting pose once so movement_control doesn't have to
        # assume where the robot is. (Start movement_control first.)
        self.init_timer = self.create_timer(STARTUP_POSE_DELAY, self.publish_initial_pose)

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
    # Movement control -> slam: start an iteration and predict
    # ------------------------------------------------------------------
    def segment_callback(self, msg):
        d = msg.data
        if len(d) < 4:
            self.get_logger().error("robot_movement_vw needs 4 values")
            return

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
        it = self.robot_iter
        if it is None:
            return

        window_start = it.start_window + SETTLE_TIME
        window_end = window_start + DELAY

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
            self.safe_ekf_step(it, 0.0, 0.0, 0.0, tags, "correction")
            it.seen_stamps.add(t_img)
            it.frames_used += 1
            it.tag_ids.update(tags)

    # ------------------------------------------------------------------
    # End of window: hand the pose back to movement
    # ------------------------------------------------------------------
    def commit(self):
        self.commit_timer = self.cancel_timer(self.commit_timer)
        it = self.robot_iter
        if it is None:
            return

        self.robot_iter = None  # tags are ignored until the next segment
        x, y, theta = self.ekf.state[:3]
        self.get_logger().info(
            f"commit: frames={it.frames_used} tags={sorted(it.tag_ids)} "
            f"pose=({x:.3f}, {y:.3f}, {np.degrees(theta):+.1f}deg)")
        self.get_logger().info(
            f"EKF state vector: {np.array2string(self.ekf.state, precision=4)}"
        )
        self.publish_pose(1.0 if it.ok else 0.0)

    # ------------------------------------------------------------------
    # slam -> movement control
    # ------------------------------------------------------------------
    def publish_initial_pose(self):
        self.init_timer = self.cancel_timer(self.init_timer)
        self.publish_pose(1.0)

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