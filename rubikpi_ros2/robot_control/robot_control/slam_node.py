"""Run a fixed-lag AprilTag EKF and publish a forward-predicted robot pose."""

from dataclasses import dataclass, field
from math import cos, isfinite, radians, sin
import re

from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
import numpy as np
import rclpy
from rclpy.clock import JumpThreshold
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from .ekf_slam import EKFSLAM, MotionNoise, TagObservation


# Store event times and the EKF cursor as exact ROS nanoseconds. Convert
# elapsed intervals to seconds when calling the numerical filter.
NANOSECONDS = 1_000_000_000
ZERO_COMMAND = (0., 0.)


@dataclass(order=True)
class Event:
    """Order pending events by timestamp, then by their receipt sequence."""

    # Sort first by timestamp, then by receipt sequence for ties. Payloads
    # are excluded from ordering because they can contain NumPy arrays.
    stamp: int
    sequence: int
    kind: str = field(compare=False)
    payload: object = field(compare=False)


class SLAMNode(Node):
    """Own the ROS interfaces and keep the EKF on a delayed timeline."""

    def __init__(self):
        """Configure the filter, subscriptions, and publication timer."""
        super().__init__('ekf_slam_node')
        # The lag controls when an event becomes eligible. The timer controls
        # when we check the queue. The maximum step bounds one integration step.
        self.lag = self._duration('processing_lag_sec', 0.12, allow_zero=True)
        period = self._duration('timer_period_sec', 0.05)
        self.max_step = self._duration('max_prediction_step_sec', 0.05)
        self.command_timeout = self._duration('command_timeout_sec', 0.15)
        self.future_tolerance = self._duration('future_tolerance_sec', 0.05, allow_zero=True)
        self.diagnostic_period = self._duration('diagnostic_period_sec', 1.0)
        # Bound both the pending event queue and remembered image identities.
        self.max_events = self.declare_parameter('max_pending_events', 256).value
        self.max_frames = self.declare_parameter('max_cached_frames', 256).value
        if self.max_events < 1 or self.max_frames < 1:
            raise ValueError('Event and frame cache limits must be positive')

        # Select the intended camera and tag frames on the shared TF topic.
        # The tag prefix is configurable for the standalone detector launch.
        camera_frame = self.declare_parameter('camera_frame', 'camera_frame').value
        self.camera_frame = camera_frame.lstrip('/')
        prefix = self.declare_parameter('tag_frame_prefix', 'tag36h11:').value
        self.tag_frame = re.compile(re.escape(prefix) + r'([0-9]+)')
        # These defaults describe the camera origin and orientation in robot
        # axes and come from the existing tag localization node.
        self.camera_translation = self._vector('camera_translation_m', [0.095, 0.0, 0.127])
        roll, pitch, yaw = map(radians, self._vector('camera_rpy_deg', [-80.0, 0.0, -90.0]))
        # Apply fixed-axis roll, pitch, then yaw to camera-frame vectors.
        rx = ((1, 0, 0), (0, cos(roll), -sin(roll)), (0, sin(roll), cos(roll)))
        ry = ((cos(pitch), 0, sin(pitch)), (0, 1, 0), (-sin(pitch), 0, cos(pitch)))
        rz = ((cos(yaw), -sin(yaw), 0), (sin(yaw), cos(yaw), 0), (0, 0, 1))
        self.camera_rotation = np.asarray(rz) @ np.asarray(ry) @ np.asarray(rx)

        # These noise defaults are starting values for rover calibration.
        sigma_forward = self.declare_parameter('sigma_forward_m', 0.05).value
        sigma_left = self.declare_parameter('sigma_left_m', 0.05).value
        self.unestimated_variance = self.declare_parameter('unestimated_variance', 1e6).value
        # Square the tag measurement standard deviations. The larger pose
        # variance is for dimensions the planar EKF does not estimate.
        variances = (sigma_forward ** 2, sigma_left ** 2, self.unestimated_variance)
        if (
            min(sigma_forward, sigma_left) <= 0
            or not all(isfinite(value) and value > 0 for value in variances)
        ):
            raise ValueError('Measurement sigmas and unestimated variance must be positive')
        self.measurement_covariance = np.diag(variances[:2])
        noise = MotionNoise(**{
            name: self.declare_parameter(name, default).value
            for name, default in vars(MotionNoise()).items()
        })

        # This saved estimate receives the camera corrections. Its time cursor
        # records how far the motion history has been integrated.
        self.ekf = EKFSLAM(noise_config=noise)
        self.clock = self.get_clock()
        self.ekf_time = self.clock.now().nanoseconds or None
        # Begin stationary. As the saved cursor passes velocity events, this
        # pair becomes the command in effect at the cursor.
        self.active_command = ZERO_COMMAND
        self.events = []
        self.sequence = 0
        self.frame_cache = set()
        # Track command freshness independently of the saved cursor. The newest
        # heartbeat can be valid while its event is still queued.
        self.latest_command_stamp = None
        self.latest_command_arrival = None
        self.fault = None
        self.clock_jumped = False
        self.late_frames = 0
        self.duplicate_frames = 0
        self.dropped_frames = 0
        self.rejected_commands = 0
        self.image_latency = None
        self.last_diagnostic = None

        # Expose the extrapolated pose for navigation and the saved landmarks for
        # inspection. The subscriptions below feed the timer through the queue.
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/slam/pose', 10)
        self.landmark_pub = None
        if self.declare_parameter('publish_landmarks', True).value:
            self.landmark_pub = self.create_publisher(MarkerArray, '/slam/landmarks', 1)
        self.create_subscription(TwistStamped, '/slam/cmd_velocity', self.velocity_callback, 10)
        self.create_subscription(TFMessage, '/tf', self.camera_callback, 100)
        # A clock-source change or discontinuity invalidates the motion timeline.
        self.jump_handle = self.clock.create_jump_callback(
            JumpThreshold(
                min_forward=Duration(seconds=1.0),
                min_backward=Duration(nanoseconds=-1),
                on_clock_change=True,
            ),
            post_callback=self._clock_jump,
        )
        self.create_timer(period / NANOSECONDS, self.timer_callback)

    def _duration(self, name, default, *, allow_zero=False):
        value = self.declare_parameter(name, default).value
        if not isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
        # Round once so event, timer and cutoff comparisons share integer units.
        nanoseconds = round(value * NANOSECONDS)
        if not allow_zero and nanoseconds == 0:
            raise ValueError(f'{name} must be at least one nanosecond')
        return nanoseconds

    def _vector(self, name, default):
        values = np.asarray(self.declare_parameter(name, default).value)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f'{name} must contain three finite values')
        return values

    def _clock_jump(self, _jump):
        # Clock callbacks can run outside the estimator's executor callbacks.
        if self.ekf_time is not None:
            self.clock_jumped = True

    def _latch_fault(self, reason):
        # Latch the first fault and clear pending work. A fresh heartbeat alone
        # cannot reconstruct missing motion, so restart is required.
        if self.fault is None:
            self.fault = reason
            self.events.clear()
            self.get_logger().error(
                f'SLAM pose unavailable: {reason}. Restart SLAM while stationary to recover.'
            )

    def _now(self):
        # Check the common ROS clock before accepting an event or integrating.
        # A backward jump behind the saved cursor invalidates the timeline.
        now = self.clock.now().nanoseconds
        if self.clock_jumped or (self.ekf_time is not None and now < self.ekf_time):
            self._latch_fault('ROS clock discontinuity')
        if now == 0 or self.fault is not None:
            return None
        if self.ekf_time is None:
            # Simulated ROS time can be zero until the first clock message.
            self.ekf_time = now
        return now

    @staticmethod
    def _stamp_ns(stamp):
        if stamp.sec < 0 or stamp.nanosec >= NANOSECONDS:
            return 0
        return stamp.sec * NANOSECONDS + stamp.nanosec

    def _enqueue(self, stamp, kind, payload):
        # Report a full queue to the caller. Dropping a frame loses one correction.
        # Dropping a command makes the motion timeline unreliable.
        if len(self.events) >= self.max_events:
            return False
        self.events.append(Event(stamp, self.sequence, kind, payload))
        self.sequence += 1
        return True

    def velocity_callback(self, msg):
        """Validate and queue a nominal command without advancing the EKF."""
        now = self._now()
        if now is None:
            return
        # Use controller publish time as the nominal command transition time.
        # Repeated commands act as heartbeats and still mark event boundaries.
        stamp = self._stamp_ns(msg.header.stamp)
        command = (msg.twist.linear.x, msg.twist.angular.z)
        if (
            stamp <= 0
            # A command older than the saved cursor would require replaying history.
            or stamp < self.ekf_time
            or stamp > now + self.future_tolerance
            or not all(map(isfinite, command))
        ):
            self.rejected_commands += 1
            self._latch_fault('invalid or late velocity command')
            return

        # An old packet must not hide a dropout or revive a stale stream.
        if self._command_is_stale(now):
            self._latch_fault('velocity command heartbeat expired')
            return
        if now - stamp > self.command_timeout:
            self.rejected_commands += 1
            self._latch_fault('velocity command arrived after its freshness limit')
            return
        if not self._enqueue(stamp, 'velocity', command):
            self.rejected_commands += 1
            self._latch_fault('pending queue overflow dropped a velocity command')
            return
        # Out-of-order commands can still be useful within the lag window, but
        # only a newer stamp advances the freshness watchdog.
        if self.latest_command_stamp is None or stamp > self.latest_command_stamp:
            self.latest_command_stamp = stamp
            self.latest_command_arrival = now

    def _command_is_stale(self, now):
        if self.latest_command_stamp is None:
            return False
        return (
            # Check age at the source and age since receipt. Delayed packets should
            # not make the stream appear current.
            now - self.latest_command_stamp > self.command_timeout
            or now - self.latest_command_arrival > self.command_timeout
        )

    def camera_callback(self, msg):
        """Queue camera-relative transforms grouped by their original image stamp."""
        now = self._now()
        if now is None:
            return
        # Read direct TF transforms from this message and group them by camera
        # and image stamp, so older cached tag poses cannot enter the frame.
        frames = {}
        for transform in msg.transforms:
            parent = transform.header.frame_id.lstrip('/')
            match = self.tag_frame.fullmatch(transform.child_frame_id.lstrip('/'))
            if parent != self.camera_frame or match is None:
                continue

            translation = transform.transform.translation
            # The translation is the tag center in optical-camera axes. Positive z
            # places the tag in front of the camera.
            p_camera = np.array((translation.x, translation.y, translation.z))
            if not np.all(np.isfinite(p_camera)) or p_camera[2] <= 0:
                continue
            # Rotate and translate the 3D point before dropping the vertical component.
            p_robot = self.camera_rotation @ p_camera + self.camera_translation
            stamp = self._stamp_ns(transform.header.stamp)
            tag_id = int(match[1])
            if tag_id > 2_147_483_647 or not np.all(np.isfinite(p_robot)):
                continue
            # Bundle all tags from this image into one event and keep the first
            # valid transform for each tag ID.
            observations = frames.setdefault((parent, stamp), {})
            observations.setdefault(
                tag_id, TagObservation(tag_id, p_robot[:2], self.measurement_covariance)
            )

        # A camera frame can be discarded without changing command validity.
        # Remember its identity only if it is accepted into the queue.
        for key, observations in frames.items():
            _, stamp = key
            if stamp <= 0 or stamp > now + self.future_tolerance:
                self.dropped_frames += 1
                continue
            if stamp < self.ekf_time:
                self.late_frames += 1
                continue
            if key in self.frame_cache:
                self.duplicate_frames += 1
                continue
            if (
                len(self.frame_cache) >= self.max_frames
                or not self._enqueue(stamp, 'camera', list(observations.values()))
            ):
                self.dropped_frames += 1
                continue

            self.frame_cache.add(key)
            self.image_latency = (now - stamp) / NANOSECONDS

    def _predict_interval(self, estimate, command, start, end):
        # Split long arcs into smaller steps and keep the exact final remainder.
        while start < end:
            step = min(self.max_step, end - start)
            estimate.predict(*command, step / NANOSECONDS)
            start += step

    def _advance_saved_ekf(self, cutoff):
        # Take eligible events only. A camera callback can arrive after a command
        # and still sort before that command by image stamp.
        ready = sorted(event for event in self.events if event.stamp <= cutoff)
        self.events = [event for event in self.events if event.stamp > cutoff]
        for event in ready:
            # The previous command remains active up to this event's timestamp.
            self._predict_interval(self.ekf, self.active_command, self.ekf_time, event.stamp)
            self.ekf_time = event.stamp
            # At the event time, set the command for the next interval or apply the
            # image measurements to the saved joint estimate.
            if event.kind == 'velocity':
                self.active_command = event.payload
            else:
                for result in self.ekf.process_observations(event.payload):
                    if result.residual is not None:
                        self.get_logger().debug(f'Tag residual (forward, left): {result.residual}')
        # Advance through the remainder even when no event was eligible. The
        # saved EKF then represents the cutoff time.
        self._predict_interval(self.ekf, self.active_command, self.ekf_time, cutoff)
        self.ekf_time = cutoff
        # Older frames will fail the timestamp check even without a cached key.
        self.frame_cache = {key for key in self.frame_cache if key[1] >= cutoff}

    def _estimate_at(self, now):
        # Replay pending commands only on a disposable copy. Camera events wait.
        estimate = self.ekf.copy()
        # Start the copy from the saved cursor and command. Local variables keep
        # the saved timeline unchanged for the next tick.
        cursor, command = self.ekf_time, self.active_command
        for event in sorted(self.events):
            if event.kind == 'velocity' and event.stamp <= now:
                self._predict_interval(estimate, command, cursor, event.stamp)
                cursor, command = event.stamp, event.payload
        # After the final known command change, predict the remaining time.
        # Leave all pending events queued for the saved filter.
        self._predict_interval(estimate, command, cursor, now)
        return estimate

    def timer_callback(self):
        """Advance the delayed map and publish a current-time pose when valid."""
        now = self._now()
        if now is None:
            return
        if self._command_is_stale(now):
            self._latch_fault('velocity command heartbeat expired')
            return

        # The lag is a duration rather than a fixed event count. At startup the
        # cutoff may precede the initial pose, so hold the saved state in place.
        cutoff = now - self.lag
        if cutoff >= self.ekf_time:
            self._advance_saved_ekf(cutoff)
        # Publish a navigation pose only after a fresh command is known at or
        # before now. The stale-stream check runs earlier in this callback.
        if self.latest_command_stamp is not None and self.latest_command_stamp <= now:
            self._publish_pose(self._estimate_at(now), now)

        if self.last_diagnostic is None or now - self.last_diagnostic >= self.diagnostic_period:
            self.last_diagnostic = now
            self._log_diagnostics(now, cutoff)
            if self.landmark_pub is not None:
                self._publish_landmarks()

    def _ros_stamp(self, nanoseconds):
        return Time(nanoseconds=nanoseconds, clock_type=self.clock.clock_type).to_msg()

    def _publish_pose(self, estimate, now):
        # The extrapolated mean and covariance both describe the robot at now.
        # Encode planar yaw as a quaternion for the ROS pose message.
        x, y, yaw = estimate.get_robot_pose()
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self._ros_stamp(now)
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = sin(yaw / 2)
        msg.pose.pose.orientation.w = cos(yaw / 2)

        # ROS covariance order is x, y, z, roll, pitch, yaw.
        # Embed the planar cross terms and use the configured variance for the
        # unestimated z, roll and pitch dimensions.
        covariance = np.eye(6) * self.unestimated_variance
        planar = (0, 1, 5)
        covariance[np.ix_(planar, planar)] = estimate.get_robot_covariance()
        msg.pose.covariance = covariance.ravel().tolist()
        self.pose_pub.publish(msg)

    def _publish_landmarks(self):
        msg = MarkerArray()
        # Landmarks come from the saved filter and carry its delayed timestamp.
        stamp = self._ros_stamp(self.ekf_time)
        for tag_id, (position, _) in self.ekf.get_landmarks().items():
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = 'apriltags'
            marker.id = tag_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.08
            marker.color.g = marker.color.a = 1.0
            msg.markers.append(marker)
        self.landmark_pub.publish(msg)

    def _log_diagnostics(self, now, cutoff):
        age = None
        if self.latest_command_stamp is not None:
            age = (now - self.latest_command_stamp) / NANOSECONDS
        covariance = self.ekf.get_robot_covariance().diagonal()
        self.get_logger().debug(
            f'ekf_time_ns={self.ekf_time} cutoff_ns={cutoff} queue={len(self.events)} '
            f'command_age_s={age} image_latency_s={self.image_latency} '
            f'late_frames={self.late_frames} duplicate_frames={self.duplicate_frames} '
            f'dropped_frames={self.dropped_frames} rejected_commands={self.rejected_commands} '
            f'robot_variance={covariance}'
        )


def main(args=None):
    """Run the node with the single-threaded ROS executor."""
    # One executor thread serializes subscriptions and timer callbacks so
    # they can safely share the queue and saved filter.
    rclpy.init(args=args)
    node = None
    try:
        node = SLAMNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
