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
        cr*cp*cy + sr*sp*sy,
    )


def quaternion_yaw(q):
    x, y, z, w = q
    return math.atan2(
        2.0 * (w*z + x*y),
        1.0 - 2.0 * (y*y + z*z),
    )


class TagLocalizationNode(Node):

    def __init__(self):
        super().__init__('tag_localization')

        self.declare_parameter('camera_frame', 'camera_frame') #tracks which camera the coordinates are coming from, here we are only processing from 1 camera but this is generalizble to if we have multiple cameras

        # Extrinsic camera pose expressed in the robot/base frame.
        self.declare_parameter(
            'base_to_camera_xyz', [0.095, 0.0, 0.127]
        )

        self.declare_parameter(
            'base_to_camera_rpy',
            [math.radians(-80), 0.0, math.radians(-90)] # The -80 degree accounts for camera tilt upwards away from robot body (pitch 10 degrees for us)
        )

        self.declare_parameter('detections_topic', '/tf')
        self.declare_parameter('pose_topic', '/robot_pose')

        self.camera_frame = str(
            self.get_parameter('camera_frame').value
        )
        

        self.tag_world = {}

        for tag_id, tag in config['tags'].items():
            x, y, z = tag['position']
            yaw = math.radians(tag['yaw_deg'])

            tag_rotation = quat_multiply(
                rpy_quaternion(0.0, 0.0, yaw),
                rpy_quaternion(math.pi / 2, 0.0, 0.0)
            )

            self.tag_world[f'tag36h11:{tag_id}'] = (
                (float(x), float(y), float(z)),
                tag_rotation,
            )

        xyz = list(
            self.get_parameter('base_to_camera_xyz').value
        )
        rpy = list(
            self.get_parameter('base_to_camera_rpy').value
        )

        if len(xyz) != 3 or len(rpy) != 3:
            raise ValueError(
                'Camera extrinsics must be XYZ and RPY triples'
            )

        # T_base_camera: camera pose expressed in the base frame.
        self.base_to_camera = (
            tuple(float(v) for v in xyz),
            rpy_quaternion(*(float(v) for v in rpy)),
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
            f'Listening to /tf; publishing world robot pose on '
            f'{self.get_parameter("pose_topic").value}'
        )

        if not self.tag_world:
            self.get_logger().warning(
                'No tag world coordinates configured. '
                'No robot pose will be published until they are set.'
            )

    def fuse_apriltag_observations(self, observations):
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
        MAX_DEVIATION = 0.20  # meters
        valid = distances <= MAX_DEVIATION

        if not np.any(valid):
            return None

        xs = xs[valid]
        ys = ys[valid]
        yaws = yaws[valid]
        distances = distances[valid]

        # Step 4: Weight observations by consistency.
        # More consistent observations receive higher weights.
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
        observations = []

        for stamped in msg.transforms:
            parent = stamped.header.frame_id.lstrip('/')
            child = stamped.child_frame_id.lstrip('/')

            if parent != self.camera_frame:
                continue
            if child not in self.tag_world:
                continue

            t = stamped.transform.translation
            r = stamped.transform.rotation

            # TF gives T_camera_tag.
            camera_to_tag = (
                (t.x, t.y, t.z),
                (r.x, r.y, r.z, r.w),
            )
            # Determine the directions of the tag's local axes
            # expressed in the camera frame.
            q = camera_to_tag[1]

            tag_x_axis = rotate_vector(q, (1.0, 0.0, 0.0))
            tag_y_axis = rotate_vector(q, (0.0, 1.0, 0.0))
            tag_z_axis = rotate_vector(q, (0.0, 0.0, 1.0))

            self.get_logger().info(
                f"{child} tag axes in camera frame:\n"
                f"  Tag +X: {tuple(round(v, 3) for v in tag_x_axis)}\n"
                f"  Tag +Y: {tuple(round(v, 3) for v in tag_y_axis)}\n"
                f"  Tag +Z: {tuple(round(v, 3) for v in tag_z_axis)}"
            )

            # get world coorindates of tag
            world_to_tag = self.tag_world[child]

            # T_world_base =
            # T_world_tag * T_tag_camera * T_camera_base
            # convert camera to robots position in the world
            tag_to_camera = transform_inverse(camera_to_tag)

            world_to_camera = transform_compose(
                self.tag_world[child],
                tag_to_camera
            )
            camera_pos = world_to_camera[0]
            self.get_logger().info(
                f"Camera world position from {child}: "
                f"x={camera_pos[0]:.3f}, "
                f"y={camera_pos[1]:.3f}, "
                f"z={camera_pos[2]:.3f}"
            )

            camera_to_base = transform_inverse(self.base_to_camera)

            world_to_base = transform_compose(
                transform_compose(world_to_tag, tag_to_camera),
                camera_to_base,
            )

            # [TAG POSE] Robot position estimated independently from this AprilTag.
            robot_x, robot_y, robot_z = world_to_base[0]
            robot_yaw = math.degrees(quaternion_yaw(world_to_base[1]))

            self.get_logger().info(
                f'[TAG POSE] {child} | '
                f'X={robot_x:.3f}m, '
                f'Y={robot_y:.3f}m, '
                f'Z={robot_z:.3f}m, '
                f'Yaw={robot_yaw:+.1f}deg'
            )

            observations.append(world_to_base)

        if not observations:
            # No configured tag was detected in this TF message.
            return

        # averaging positions established by each april tag seen 
        x, y, yaw = self.fuse_apriltag_observations(observations)

        out = Pose2D()
        out.x = x
        out.y = y
        out.theta = yaw

        self.pose_pub.publish(out)

        self.get_logger().info(
            f'Robot world pose: x={x:.3f} m, '
            f'y={y:.3f} m, yaw={yaw:.3f} rad'
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

