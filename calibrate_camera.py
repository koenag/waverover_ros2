
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Checkerboard: internal corners, square size in meters
BOARD = (9, 6)
SQUARE = 0.022
TARGET = 50

objp = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE


class CameraCalibrator(Node):
    def __init__(self):
        super().__init__('camera_calibrator')
        self.bridge = CvBridge()
        self.objects = []
        self.images = []
        self.last_capture = 0
        self.last_corners = None
        self.size = None

        self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        print("Move the checkerboard around at different angles and distances.")

    def callback(self, msg):
        if len(self.images) >= TARGET:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, BOARD)
        if not found:
            return

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # Avoid capturing nearly identical views.
        if time.monotonic() - self.last_capture < 2:
            return
        if self.last_corners is not None:
            movement = np.mean(np.linalg.norm(
                corners.reshape(-1, 2) - self.last_corners.reshape(-1, 2),
                axis=1
            ))
            if movement < 25:
                return

        self.objects.append(objp.copy())
        self.images.append(corners)
        self.last_corners = corners.copy()
        self.last_capture = time.monotonic()
        print(f"Captured {len(self.images)}/{TARGET}", flush=True)


def main():
    rclpy.init()
    node = CameraCalibrator()

    try:
        while rclpy.ok() and len(node.images) < TARGET:
            rclpy.spin_once(node, timeout_sec=0.5)

        if len(node.images) < TARGET:
            return

        error, K, D, _, _ = cv2.calibrateCamera(
            node.objects, node.images, node.size, None, None
        )
        print(f"Calibration RMS error: {error:.3f} pixels")

        # Save using the repository's OpenCV YAML format.
        path = '/home/ubuntu/waverover_ros2/camera_parameter_calibrated.yaml'
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        fs.write('image_width', node.size[0])
        fs.write('image_height', node.size[1])
        fs.write('camera_name', 'camera')
        fs.write('camera_matrix', K)
        fs.write('distortion_model', 'plumb_bob')
        fs.write('distortion_coefficients', D.reshape(1, -1))
        fs.write('rectification_matrix', np.eye(3))
        fs.write('projection_matrix', np.hstack((K, np.zeros((3, 1)))))
        fs.release()
        print(f"Saved calibration to {path}")

    except KeyboardInterrupt:
        print("Calibration interrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

