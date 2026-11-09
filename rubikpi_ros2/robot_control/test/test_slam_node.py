"""Check timestamp ordering and fail-closed pose publication."""

from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest


def _command(stamp, v):
    """Build a stamped controller heartbeat using exact integer time."""
    from geometry_msgs.msg import TwistStamped

    msg = TwistStamped()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 10**9)
    msg.twist.linear.x = v
    return msg


@pytest.fixture
def slam():
    """Run a ROS node with a deterministic clock and captured poses."""
    rclpy = pytest.importorskip('rclpy')
    from robot_control.slam_node import SLAMNode

    rclpy.init()
    node = SLAMNode()
    # Keep the real ROS clock type for message conversion.
    fake_clock = SimpleNamespace(
        clock_type=node.clock.clock_type,
        now=lambda: SimpleNamespace(nanoseconds=fake_clock.nanoseconds),
        nanoseconds=10_000_000_000,
    )
    node.clock = fake_clock
    node.ekf_time = fake_clock.nanoseconds
    published = []
    node.pose_pub = SimpleNamespace(publish=published.append)
    try:
        yield node, fake_clock, published
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_delayed_camera_and_live_copy(slam):
    """Apply a delayed camera frame before a newer command, exactly once."""
    from geometry_msgs.msg import TransformStamped
    from tf2_msgs.msg import TFMessage

    node, clock, published = slam
    start = clock.nanoseconds
    node.velocity_callback(_command(start, 0.2))
    clock.nanoseconds = start + 50_000_000
    node.velocity_callback(_command(clock.nanoseconds, 0.0))

    # A frame stamped during the drive arrives after the stop.
    tf = TransformStamped()
    tf.header.frame_id = 'camera_frame'
    tf.header.stamp.sec = 10
    tf.header.stamp.nanosec = 25_000_000
    tf.child_frame_id = 'tag36h11:4'
    tf.transform.translation.z = 2.0
    frame = TFMessage(transforms=[tf, tf])
    clock.nanoseconds = start + 85_000_000
    node.camera_callback(frame)
    node.camera_callback(frame)

    clock.nanoseconds = start + 100_000_000
    node.velocity_callback(_command(clock.nanoseconds, 0.0))
    node.timer_callback()
    assert node.ekf_time == start
    assert_allclose(node.ekf.get_robot_pose(), [0., 0., 0.])
    assert_allclose(published[-1].pose.pose.position.x, 0.01)

    clock.nanoseconds = start + 200_000_000
    node.velocity_callback(_command(clock.nanoseconds, 0.0))
    node.timer_callback()
    assert node.ekf_time == start + 80_000_000
    assert node.duplicate_frames == 1
    assert_allclose(node.ekf.get_robot_pose(), [0.01, 0., 0.])
    # The tag was initialized at x=0.005, halfway through the drive.
    expected = [0.005 + 0.095 + 2 * np.sin(np.deg2rad(80)), 0.]
    assert_allclose(node.ekf.get_landmarks()[4][0], expected, atol=1e-12)
    assert published[-1].header.stamp.nanosec == 200_000_000
    assert_allclose(published[-1].pose.pose.position.x, 0.01)
    assert len(published[-1].pose.covariance) == 36


def test_stale_command_latches_pose_fault(slam):
    """A command arriving after a heartbeat gap cannot silently restore pose."""
    node, clock, published = slam
    node.velocity_callback(_command(clock.nanoseconds, 0.2))
    node.timer_callback()
    assert len(published) == 1
    clock.nanoseconds += node.command_timeout + 1
    node.velocity_callback(_command(clock.nanoseconds, 0.0))
    node.timer_callback()
    assert node.fault == 'velocity command heartbeat expired'
    assert len(published) == 1
    assert not node.events


def test_late_command_latches_pose_fault(slam):
    """Do not apply a command retroactively once the saved cursor passes it."""
    node, clock, published = slam
    start = clock.nanoseconds
    clock.nanoseconds += node.lag + 10_000_000
    node.timer_callback()
    node.velocity_callback(_command(start, 0.2))
    assert node.rejected_commands == 1
    assert node.fault is not None
    assert not published
