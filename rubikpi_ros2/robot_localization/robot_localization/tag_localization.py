
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Pose2D

import os
import yaml
from ament_index_python.packages import get_package_share_directory

config_path = os.path.join(
    get_package_share_directory('robot_localization'),
    'config',
    'apriltag_map.yaml'
)

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_inverse(q):
    x, y, z, w = q
    norm_sq = x*x + y*y + z*z + w*w
    if norm_sq < 1e-12:
        raise ValueError('Cannot invert a zero quaternion')
    return (-x/norm_sq, -y/norm_sq, -z/norm_sq, w/norm_sq)


def rotate_vector(q, vector):
    rotated = quat_multiply(
        quat_multiply(q, (*vector, 0.0)),
        quat_inverse(q),
    )
    return rotated[:3]


def transform_compose(a, b):
    """Compose transforms: T_A_C = T_A_B * T_B_C."""
    ta, qa = a
    tb, qb = b
    offset = rotate_vector(qa, tb)
    return (
        (ta[0] + offset[0], ta[1] + offset[1], ta[2] + offset[2]),
        quat_multiply(qa, qb),
    )


def transform_inverse(transform):
    translation, rotation = transform
    inverse_rotation = quat_inverse(rotation)
    inverse_translation = rotate_vector(
        inverse_rotation,
        (-translation[0], -translation[1], -translation[2]),
    )
    return inverse_translation, inverse_rotation



def rpy_quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    return (
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy - sr*sp*sy,
    )


# Gets yaw from quarternion.
def quaternion_yaw(q):
    x, y, z, w = q
    return math.atan2(
        2.0 * (w*z + x*y),
        1.0 - 2.0 * (y*y + z*z),
    )


class TagLocalizationNode(Node):

    def __init__(self):
        super().__init__('tag_localization')

        self.declare_parameter('camera_frame', 'camera_frame') #scales to if we have multiple cameras
        self.declare_parameter('detections_topic', '/tf')
        self.declare_parameter('pose_topic', '/robot_pose')

        # sets the camera frame that we are accesssing (in this case there's only 1 camera)
        self.camera_frame = str(
            self.get_parameter('camera_frame').value
        )

        # building the extrinsic matrix
        self.translation_coeffs = [0.095, 0.0, 0.127]
        self.rotation_coeffs = [math.radians(-80), 0.0, math.radians(-90)]
        self.extrinsic_matrix = (
            tuple(float(v) for v in self.translation_coeffs),
            rpy_quaternion(self.rotation_coeffs[0], self.rotation_coeffs[1], self.rotation_coeffs[2])
        )
        
    
        self.tag_in_world_coordinates= {}
        for tag_id, tag in config['tags'].items():
            x, y, z = tag['position']
            yaw = math.radians(tag['yaw_deg'])

            # the roll is to consider that it's mounted on the wall (rotation in x axis)
            # Z+ is facing outside of the april tag, Y+ is facing down, X+ is facing right
            tag_rotation = quat_multiply(
                rpy_quaternion(0.0, 0.0, yaw),
                rpy_quaternion(math.pi / 2, 0.0, 0.0)
            )

            self.tag_in_world_coordinates[f'tag36h11:{tag_id}'] = (
                (float(x), float(y), float(z)),
                tag_rotation,
            )


        self.pose_pub = self.create_publisher(
            Pose2D,
            str(self.get_parameter('pose_topic').value),
            10,
        )

        self.tf_sub = self.create_subscription(
            TFMessage,
            str(self.get_parameter('detections_topic').value),
            self.tf_callback,
            10,
        )

        self.get_logger().info(
            f'[START] Localization initialized | '
            f'Camera={self.camera_frame} | '
            f'Configured tags={len(self.tag_in_world_coordinates)} | '
            f'Publishing to {self.get_parameter("pose_topic").value}'
        )

        if not self.tag_in_world_coordinates:
            self.get_logger().warning(
                '[CONFIG] No AprilTag world coordinates configured'
            )

    def fuse_apriltag_observations(self, observations):
        '''
        takes in a list of potential robot poses [x, y, theta]
        computes a weighted average of the potential poses to approxiamate the true pose
        '''
        if not observations:
            return None

        # Extract robot pose estimates from individual AprilTags.
        xs = np.array([pose[0][0] for pose in observations])
        ys = np.array([pose[0][1] for pose in observations])
        yaws = np.array([
            quaternion_yaw(pose[1]) for pose in observations
        ])

        # Step 1: Find the median position.
        median_x = np.median(xs)
        median_y = np.median(ys)

        # Step 2: Measure disagreement with the median.
        distances = np.hypot(
            xs - median_x,
            ys - median_y
        )

        # Step 3: Reject large positional outliers.
        MAX_DEVIATION = 0.20
        valid = distances <= MAX_DEVIATION

        if not np.any(valid):
            return None

        xs = xs[valid]
        ys = ys[valid]
        yaws = yaws[valid]
        distances = distances[valid]

        # Step 4: Weight observations by consistency.
        weights = 1.0 / (0.05 + distances)
        weights /= np.sum(weights)

        # Step 5: Fuse position.
        x = float(np.sum(weights * xs))
        y = float(np.sum(weights * ys))

        # Step 6: Fuse orientation using circular averaging.
        yaw = math.atan2(
            np.sum(weights * np.sin(yaws)),
            np.sum(weights * np.cos(yaws))
        )

        return x, y, yaw

    def tf_callback(self, msg):
        '''
        Recieves AprilTag coordinates from tf node
        Performs transformation:
            Camera --> World
                1. Where is the camera from the tag
                2. Where is the tag from the world
                This tells us where is the camera from the world
                
                tag_to_camera is camera coordinates relative to tag reference frame
                world_to_tag is tag coordinates relative to world reference frame

                Camera coordinates in world refernce is: world_to_tag * tag_to_camera
                
        '''
        observed_poses = []
        tag_details = []

        for stamped in msg.transforms:
            parent = stamped.header.frame_id.lstrip('/')
            child = stamped.child_frame_id.lstrip('/')

            if parent != self.camera_frame:
                continue
            if child not in self.tag_in_world_coordinates:
                continue

            t = stamped.transform.translation
            r = stamped.transform.rotation

            camera_to_tag = (
                (t.x, t.y, t.z),
                (r.x, r.y, r.z, r.w),
            ) # tag coordinates relative to camera reference frame
            world_to_tag = self.tag_in_world_coordinates[child] # tag coordinates relative to world reference frame

            # find camera coordinates relative to world reference frame
            world_to_camera = transform_compose(
                world_to_tag, 
                transform_inverse(camera_to_tag)
            )
            camera_to_center_of_robot = transform_compose(
                world_to_camera, 
                transform_inverse(self.extrinsic_matrix)
            )

            observed_poses.append(camera_to_center_of_robot)

            # Save individual tag estimates for diagnostic logging.
            robot_x, robot_y, robot_z = camera_to_center_of_robot[0]
            robot_yaw = math.degrees(
                quaternion_yaw(camera_to_center_of_robot[1])
            )

            tag_details.append(
                (
                    child,
                    robot_x,
                    robot_y,
                    robot_z,
                    robot_yaw
                )
            )

        # No configured AprilTags in this TF message.
        if not observed_poses:
            return

        # Log all tag estimates together for easier comparison.
        self.get_logger().info(
            f'[DETECTION] {len(observed_poses)} AprilTag(s) received'
        )

        for child, x, y, z, yaw in tag_details:
            self.get_logger().info(
                f'[TAG {child}] '
                f'X={x:.3f}m | '
                f'Y={y:.3f}m | '
                f'Z={z:.3f}m | '
                f'Yaw={yaw:+.1f}deg'
            )

        # collate individual AprilTag observations to estimate actual robot pose
        averaged_robot_pose = self.fuse_apriltag_observations(observed_poses)

        # Gracefully handle rejected/invalid fusion results.
        if averaged_robot_pose is None:
            self.get_logger().warning(
                '[FUSION FAILED] All AprilTag observations '
                'were rejected (position disagreement > 0.20m). '
                'Skipping this pose update.'
            )
            return

        x, y, yaw = averaged_robot_pose
        out = Pose2D()
        out.x = x
        out.y = y
        out.theta = yaw

        self.pose_pub.publish(out)

        self.get_logger().info(
            f'[PUBLISHED POSE] '
            f'X={x:.3f}m | '
            f'Y={y:.3f}m | '
            f'Yaw={math.degrees(yaw):+.1f}deg'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TagLocalizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
