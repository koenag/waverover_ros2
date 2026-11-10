#!/usr/bin/env python3
"""
Calibrate R for AprilTag measurements (single file, 4x4 matrices).

Robot + tag stationary. Run once per distance / bearing / angle:

  python3 calibrate_R.py --ros-args -p label:=d1.0 -p n_frames:=150

TF gives camera_frame -> tag  =  tag_in_camera_system (tag pose in camera frame, no inverse).
Extrinsic is camera pose in robot frame = camera_in_robot_system (no inverse).
Tag in robot frame:  tag_in_robot_system = camera_in_robot_system @ tag_in_camera_system   ->  z = [forward, left]
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_to_matrix(x, y, z, w):
    n = x*x + y*y + z*z + w*w
    s = 2.0 / n
    return np.array([
        [1 - s*(y*y + z*z), s*(x*y - z*w),     s*(x*z + y*w)],
        [s*(x*y + z*w),     1 - s*(x*x + z*z), s*(y*z - x*w)],
        [s*(x*z - y*w),     s*(y*z + x*w),     1 - s*(x*x + y*y)],
    ])


def make_T(R, t):
    # R is the quarterntion rotation, t is the translation so x,y,z where the tag is from the origin
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# this is how the camera coordinate origin is rotated and offset from the robots frame that is from the center of the robot
camera_in_robot_system = make_T(
    rpy_to_matrix(math.radians(-80), 0.0, math.radians(-90)),
    [0.095, 0.0, 0.127],
)


class CalibrateR(Node):
    def __init__(self):
        super().__init__('calibrate_R')
        self.declare_parameter('camera_frame', 'camera_frame')
        self.declare_parameter('n_frames', 150)
        self.declare_parameter('label', 'trial')
        self.declare_parameter('true_fwd', 1.0)
        self.declare_parameter('true_left', 0.0)

        self.true_fwd = float(self.get_parameter('true_fwd').value)
        self.true_left = float(self.get_parameter('true_left').value)
        self.camera_frame = self.get_parameter('camera_frame').value
        self.n = int(self.get_parameter('n_frames').value)
        self.label = self.get_parameter('label').value

        self.data = {}        # tag -> list of [fwd, left, up]
        self.last_stamp = {}  # tag -> last stamp (skip duplicates)
        self.done = False
        self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)
        self.get_logger().info(f'collecting {self.n} frames, camera_frame={self.camera_frame}')

    def tf_callback(self, msg):
        for s in msg.transforms:
            if s.header.frame_id.lstrip('/') != self.camera_frame:
                continue
            tag = s.child_frame_id.lstrip('/')
            if not tag.startswith('tag'):
                continue
            stamp = (s.header.stamp.sec, s.header.stamp.nanosec)
            if self.last_stamp.get(tag) == stamp:
                continue
            self.last_stamp[tag] = stamp

            # tag_in_camera_system: tag pose in camera frame, straight from TF
            t, q = s.transform.translation, s.transform.rotation
            tag_in_camera_system = make_T(quat_to_matrix(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])

            # tag_in_robot_system = camera_in_robot_system @ tag_in_camera_system
            tag_in_robot_system = camera_in_robot_system @ tag_in_camera_system
            # the x,y,z of the tag in the camera coordinates
            fwd, left, up = tag_in_robot_system[:3, 3]
            error_fwd = self.true_fwd - fwd
            error_left = self.true_left - left
            self.data.setdefault(tag, []).append([error_fwd,error_left])

            n = len(self.data[tag])
            if n % 25 == 0:
                self.get_logger().info(f'{tag}: {n}/{self.n}')
            if n >= self.n:
                self.done = True

    def report(self):
        worst = np.zeros(2)

        for tag, pts in sorted(self.data.items()):
            P = np.array(pts)

            if len(P) < 10:
                print(f'{tag}: only {len(P)} frames, skipped')
                continue

            E = P[:, :2]  # [error_fwd, error_left]

            R = np.cov(E.T, ddof=1)
            R_mse = (E.T @ E) / len(E)
            bias = E.mean(axis=0)

            lag1 = [
                np.corrcoef(E[:-1, i], E[1:, i])[0, 1]
                for i in range(2)
            ]

            true_range = np.hypot(self.true_fwd, self.true_left)

            print(f'\n[{self.label}] {tag} N={len(E)} range={true_range:.3f} m')
            print(f'  mean error [fwd,left] = {bias.round(4)}')
            print(f'  std [fwd,left] = {np.sqrt(np.diag(R)).round(4)}')
            print(f'  lag1 autocorr = {np.round(lag1, 2)}')
            print(f'  R_full =\n{R}')
            print(f'  R_mse =\n{R_mse}')
            print(f'  R_diag = np.diag([{R[0,0]:.3e}, {R[1,1]:.3e}])  # m^2')

            worst = np.maximum(worst, np.diag(R))

            np.savetxt(
                f'R_{self.label}_{tag.replace(":", "_")}.csv',
                P,
                delimiter=',',
                header='error_fwd,error_left'
            )

        print(f'\nworst-case diag this trial: '
            f'np.diag([{worst[0]:.3e}, {worst[1]:.3e}])')
def main():
    rclpy.init()
    node = CalibrateR()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.report()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

