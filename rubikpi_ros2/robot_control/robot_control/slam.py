"""slam.py: coordinator between movement_control and the EKF.

The EKF is a plain Python class (ekf.py, class KalmanEKF) that this node
calls directly, so there are no ekf topics.

  robot_movement_vw    (movement -> slam)
      [seq, v, w, dt, stop_time, end_of_delay]
      Absolute ROS-clock times in seconds. Tags whose image stamp is in
      [stop_time, end_of_delay] are accepted.

  verified_pose_update (slam -> movement)
      [seq, ok, x, y, theta, n_landmarks, id1, mx1, my1, ...]
      ok = 1.0 normal, 0.0 if the EKF update failed (pose is then the
      last known pose)

At end_of_delay, slam sends ONE batch to the EKF (v, w, dt, tags averaged
per tag id; the tag dict may be empty) and publishes the result.
"""

import numpy as np
import rclpy

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

from robot_control.kalman_filter_code import KalmanEKF


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class RobotIter:
    """Holds v, w and the tags seen during one stop-and-go iteration.

    Created when movement_control publishes robot_movement_vw and set to
    None once the pose has been published back. Tags are only collected
    while one exists."""

    def __init__(self, seq, v, w, dt, end_of_delay, stop_time):
        self.seq = seq
        self.v = v
        self.w = w
        self.dt = dt
        self.end_of_delay = end_of_delay
        self.stop_time = stop_time
        self.tags = {}  # tag_id -> list of robot-frame [x, y] detections


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

        # Diagnostics
        self.n_tags_accepted = 0
        self.n_tags_rejected_time = 0

        self.create_subscription(
            Float64MultiArray, "robot_movement_vw", self.segment_callback, 10)
        self.create_subscription(
            TFMessage, "/tf", self.tf_callback, 50)

        self.pose_pub = self.create_publisher(
            Float64MultiArray, "verified_pose_update", 10)

        # Initial pose (origin) so movement_control has something to start from.
        self.init_timer = self.create_timer(2.0, self.publish_initial_pose)

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

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def publish_initial_pose(self):
        self.init_timer = self.cancel_timer(self.init_timer)
        self.publish_pose(seq=0, ok=1.0)

    # ------------------------------------------------------------------
    # Movement control -> slam: start an iteration
    # ------------------------------------------------------------------
    def segment_callback(self, msg):
        d = msg.data
        if len(d) < 6:
            self.get_logger().error("robot_movement_vw needs 6 values")
            return

        self.robot_iter = RobotIter(
            seq=int(d[0]), v=float(d[1]), w=float(d[2]), dt=float(d[3]),
            stop_time=float(d[4]),
            end_of_delay=float(d[5]),
        )

        # At end_of_delay, send whatever we have (0 or more tags).
        wait = max(0.001, self.robot_iter.end_of_delay - self.now_sec())
        # call invoke_ekf only after the delay
        self.commit_timer = self.create_timer(wait, self.invoke_ekf) # why do you need this why can you just call commit_calback as a funciton

    # ------------------------------------------------------------------
    # Detector -> slam: collect tags while an iteration exists
    # ------------------------------------------------------------------
    def tf_callback(self, msg):
        it = self.robot_iter
        if it is None:
            return

        for stamped in msg.transforms:
            parent = stamped.header.frame_id.lstrip("/")
            child = stamped.child_frame_id.lstrip("/")

            if parent != self.camera_frame:
                continue
            if not child.startswith("tag36h11:"):
                continue

            # Ignore tags captured before the previous segment stopped or
            # after the delay ended (image stamp, not arrival time).
            t_img = stamp_to_sec(stamped.header.stamp)
            if not (it.stop_time <= t_img <= it.end_of_delay):
                self.n_tags_rejected_time += 1
                continue

            try:
                tag_id = int(child.split(":", 1)[1])
            except ValueError:
                continue

            t = stamped.transform.translation
            tag_in_camera = np.array([t.x, t.y, t.z])
            if not np.all(np.isfinite(tag_in_camera)):
                continue

            tag_in_robot = (
                self.camera_translation
                + self.camera_rotation @ tag_in_camera
            )
            it.tags.setdefault(tag_id, []).append(tag_in_robot[:2])
            self.n_tags_accepted += 1

    #At End of Delay, send all the collected tags and the current robot v,w to EKF
    def invoke_ekf(self):
        """ """
        self.commit_timer = self.cancel_timer(self.commit_timer)
        it = self.robot_iter
        if it is None:
            return

        # One averaged measurement per tag (repeated frames of a static
        # scene are correlated, so they are not applied separately).
        tags = {
            tag_id: np.mean(np.array(xy_list), axis=0)
            for tag_id, xy_list in it.tags.items()
        }
  
        try:
            self.ekf.update_state(it.v, it.w, it.dt, tags)
        except Exception as e:  # e.g. LinAlgError; never leave movement hanging
            self.get_logger().error(f"ekf update failed for seq={it.seq}: {e}")
   

        self.robot_iter = None  # clear: tags are ignored until the next segment

    # ------------------------------------------------------------------
    # slam -> movement control
    # ------------------------------------------------------------------
    def publish_pose(self, seq, ok):
        """Send the state vector (landmarks and robot pose) to movement control"""
        pose = self.ekf.state[:3]
        landmarks = self.landmarks_flat()
        out = Float64MultiArray()
        out.data = (
            [float(seq), float(ok), float(pose[0]), float(pose[1]),
             float(pose[2]), float(len(self.ekf.landmark_map))]
            + landmarks
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