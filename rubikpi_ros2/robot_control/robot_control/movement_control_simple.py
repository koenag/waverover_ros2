import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray


# Motion settings
INTRINSTIC_CALIBRATION_MATRIIX = np.array([
    [0.565034, 0.733764],
    [-4.08242, 3.82298]
])
MAX_MOTOR_COMMAND = 0.5
DRIVE_SPEED = 0.2
TURN_SPEED = 3.0
POSITION_TOLERANCE = 0.1
HEADING_TOLERANCE = 0.2
MAX_DRIVE_SEGMENT = 0.20

CALLBACK_FREQUENCY = 0.05

# Must match the EKF's initial state (zeros) and the frame the waypoints are in.
OFFSET = (0.0, 0.0, 0.0)

DELAY = 1.0          # seconds stopped after a burst so slam can collect tags
STARTUP_WAIT = 3.0   # seconds to let slam come up before the first burst


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class MovementController(Node):
    def __init__(self):
        '''
        Publishes L,R to motor_commands
        Publishes [seq, v, w, dt, stop_time, end_of_delay] to robot_movement_vw
        Subscribes to verified_pose_update from slam
        '''
        super().__init__('movement_controller_node')

        # Motors take float32 [left, right]
        self.publisher = self.create_publisher(
            Float32MultiArray, 'motor_commands', 10)

        # Slam topics are float64 (absolute ROS times don't fit in float32)
        self.slam_publisher = self.create_publisher(
            Float64MultiArray, 'robot_movement_vw', 10)
        self.create_subscription(
            Float64MultiArray, 'verified_pose_update', self.update_pose, 10)

        with open('./waypoint_2.txt') as file:
            self.waypoints = [
                (
                    x + OFFSET[0],
                    y + OFFSET[1],
                    theta + OFFSET[2]
                )
                for line in file if line.strip()
                    for x, y, theta in [map(float, line.split(','))]
            ]

        self.waypoint_index = 1  # Waypoint 0 is the starting position

        self.pose = OFFSET           # latest pose from slam (x, y, theta)
        self.landmarks = {}          # latest landmarks from slam
        self.pose_time = time.monotonic()  # when self.pose was last set
        self.used_pose_time = 0.0          # pose_time the planner last used

        # Wait for slam to come up before the first burst
        self.pause_until = time.monotonic() + STARTUP_WAIT

        # (linear, requested angular, actual angular, start time, duration,
        #  starting pose, description)
        self.movement = None

        # Slam handshake
        self.seq_num = 0             # id of the last segment sent to slam
        self.waiting_slam = False    # True from the end of a burst until slam replies

        self.create_timer(CALLBACK_FREQUENCY, self.move)

        # LOG: Controller initialization
        self.get_logger().info(
            f'[START] {len(self.waypoints)} waypoints loaded | '
            f'Drive speed={DRIVE_SPEED:.2f}m/s | '
            f'Turn speed={TURN_SPEED:.2f}rad/s'
        )

    def ros_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def update_pose(self, msg):
        '''
        Slam's reply: [seq, ok, x, y, theta, n, id, mx, my, ...]
        Only accepted while we are waiting, and only for the segment we sent.
        Anything else (a late reply, slam's startup pose) is ignored.
        '''
        d = msg.data
        seq, ok = int(d[0]), d[1]
        if not self.waiting_slam or seq != self.seq_num:
            self.get_logger().warning(f"Robot pose was sent at {seq}, but now is at {self.seq_num}. Ignoring EKF localization")
            return
        
        if ok:
            self.pose = (d[2], d[3], d[4])
        else:
            self.get_logger().warn('slam update failed; keeping last pose')

        num_detected_landmarks = int(d[5])
        self.landmarks = {int(d[6 + 3*k]): (d[7 + 3*k], d[8 + 3*k]) for k in range(num_detected_landmarks)}
        self.waiting_slam = False            # True if we are in the delay period waiting for SLAM to give a localization else if we are moving is False
        self.get_logger().warning(f"Got a pose from SLAM {self.pose}. Will allow robot movement now")
        self.pose_time = time.monotonic()    # time when last pose came in is now

    def command(self, linear_speed, angular_speed):
        left, right = np.linalg.solve(
            INTRINSTIC_CALIBRATION_MATRIIX, [linear_speed, angular_speed])

        scale = max(
            1.0,
            abs(left) / MAX_MOTOR_COMMAND,
            abs(right) / MAX_MOTOR_COMMAND
        )

        self.publisher.publish(Float32MultiArray(
            data=[float(left / scale), float(right / scale)]))

        return linear_speed / scale, angular_speed / scale

    def create_movement(self, linear_speed, angular_speed, amount, starting_pose, movement_description):

        # publish the v,w before move callback starts the distance checking so this is the true start time of the motion
        actual_linear, actual_angular = self.command(
            linear_speed, angular_speed)

        actual_speed = abs(
            actual_linear if linear_speed != 0 else actual_angular)

        duration = amount / max(actual_speed, 1e-6)
        movement_start_time = time.monotonic()
        self.movement = (
            actual_linear, actual_angular, movement_start_time,
            duration, movement_description)

        # LOG: Movement command and calibration output
        left, right = np.linalg.solve(
            INTRINSTIC_CALIBRATION_MATRIIX, [linear_speed, angular_speed])

        scale = max(
            1.0,
            abs(left) / MAX_MOTOR_COMMAND,
            abs(right) / MAX_MOTOR_COMMAND
        )

        self.get_logger().info(
            f'[MOVE START] {movement_description} | '
            f'Amount={amount:.3f}{"m" if linear_speed != 0 else "rad"} | '
            f'Duration={duration:.2f}s | '
            f'v={actual_linear:+.3f}m/s | '
            f'w={actual_angular:+.3f}rad/s | '
            f'L={left / scale:+.3f}, R={right / scale:+.3f}'
        )

    def move(self):
        '''
        Burst of open-loop movement, then stop and wait for slam's pose,
        then plan the next burst from that pose.
        '''
        now = time.monotonic()

        # Debug check make sure if we have a self.movement we are not delaying its start time
        if self.waiting_slam: 
            assert not self.movement 

        # Stopped after a burst: wait until slam replies (update_pose sets waiting to false) OR  Startup wait / pause after reaching a waypoint
        if self.waiting_slam or (self.pause_until and now < self.pause_until):
            self.command(0.0, 0.0)
            return

        # In a burst: keep driving, then stop and report it to slam
        if self.movement:
            (linear_speed, angular_rotation, movement_start_time,
             duration, movement_description) = self.movement

            if now < movement_start_time + duration:
                self.command(linear_speed, angular_rotation)
                return

            self.command(0.0, 0.0)
            dt = now - movement_start_time    # how long the motors were on
            stop_time = self.ros_now()        # absolute time, must be ROS time

            self.seq_num += 1                 # once, before publishing
            self.slam_publisher.publish(Float64MultiArray(data=[
                float(self.seq_num), linear_speed, angular_rotation, dt,
                stop_time, stop_time + DELAY]))

            self.movement = None
            self.used_pose_time = self.pose_time
            self.waiting_slam = True

            self.get_logger().info(
                f'[MOVE END] {movement_description} completed | '
                f'sent seq={self.seq_num}, waiting for slam pose'
            )
            return

        # All waypoints completed.
        if self.waypoint_index >= len(self.waypoints):
            self.command(0.0, 0.0)
            return

        # Plan the next burst whenever there is a pose we have not used yet
        if self.pose is not None and self.pose_time > self.used_pose_time:
            self.used_pose_time = self.pose_time

            current_x, current_y, current_heading = self.pose
            goal_x, goal_y, goal_heading = self.waypoints[self.waypoint_index]

            dx = goal_x - current_x
            dy = goal_y - current_y
            distance_remaining = math.hypot(dx, dy)

            desired_heading_to_reach_eos = math.atan2(dy, dx)
            heading_error = wrap_angle(desired_heading_to_reach_eos - current_heading)
            final_heading_error = wrap_angle(goal_heading - current_heading)

            # LOG: Current pose and target waypoint
            self.get_logger().info(
                f'[POSE] WP {self.waypoint_index} | '
                f'Robot=({current_x:.3f}, {current_y:.3f}, '
                f'{math.degrees(current_heading):+.1f}deg) | '
                f'Target=({goal_x:.3f}, {goal_y:.3f}, '
                f'{math.degrees(goal_heading):+.1f}deg)'
            )

            # LOG: Distance and heading errors used for planning
            self.get_logger().info(
                f'[ERROR] Distance={distance_remaining:.3f}m | '
                f'Desired direction={math.degrees(desired_heading_to_reach_eos):+.1f}deg | '
                f'Heading error={math.degrees(heading_error):+.1f}deg | '
                f'Final heading error={math.degrees(final_heading_error):+.1f}deg'
            )

            # First check if you're close to the end of the segment in distance, then in heading.
            # --> If you're close enough in both, mark the waypoint as complete and move to the next one.
            # --> If you're close in distance but not heading, turn to correct your heading.
            # --> If you're not close in distance but are facing the right way, drive forward.
            # --> If you're not facing the right way, turn to correct your heading.
            if distance_remaining <= POSITION_TOLERANCE:
                if abs(final_heading_error) <= HEADING_TOLERANCE:

                    # LOG: Waypoint successfully completed
                    self.get_logger().info(
                        f'[WAYPOINT REACHED] {self.waypoint_index} | '
                        f'Position error={distance_remaining:.3f}m | '
                        f'Heading error={math.degrees(final_heading_error):+.1f}deg'
                    )
                    self.pause_until = time.monotonic() + 1.0
                    self.waypoint_index += 1
                    # No movement means no new slam pose, so mark the current
                    # pose as new to let the planner run again after the pause.
                    self.pose_time = time.monotonic()

                    if self.waypoint_index < len(self.waypoints):
                        self.get_logger().info(
                            f'[NEXT WAYPOINT] Moving to WP {self.waypoint_index}: '
                            f'{self.waypoints[self.waypoint_index]}'
                        )
                    else:
                        self.get_logger().info('[DONE] All waypoints completed')

                else:
                    # LOG: Position reached, but final heading is incorrect
                    self.get_logger().info(
                        f'[DECISION] FINAL TURN | '
                        f'Position reached, correcting heading by '
                        f'{math.degrees(final_heading_error):+.1f}deg'
                    )

                    self.create_movement(
                        0.0,
                        math.copysign(TURN_SPEED, final_heading_error),
                        abs(final_heading_error), self.pose, 'FINAL TURN'
                    )

            # facing the wrong way: rotate to correct the heading
            elif abs(heading_error) > HEADING_TOLERANCE:

                # LOG: Robot must align with target before driving
                self.get_logger().info(
                    f'[DECISION] TURN | '
                    f'Heading error={math.degrees(heading_error):+.1f}deg | '
                    f'Direction={"LEFT/CCW" if heading_error > 0 else "RIGHT/CW"}'
                )

                self.create_movement(
                    0.0,
                    math.copysign(TURN_SPEED, heading_error),
                    abs(heading_error), self.pose, 'TURN'
                )

            # facing the right way, not yet at the waypoint: drive forward
            else:
                speed = DRIVE_SPEED

                # LOG: Robot aligned sufficiently to drive
                self.get_logger().info(
                    f'[DECISION] DRIVE | '
                    f'Distance={distance_remaining:.3f}m | '
                    f'Speed={speed:.3f}m/s | '
                    f'Heading error={math.degrees(heading_error):+.1f}deg'
                )

                self.create_movement(
                    speed,
                    0.0,
                    min(distance_remaining, MAX_DRIVE_SEGMENT),
                    self.pose,
                    'DRIVE'
                )

            return

        # Nothing to do: stay stopped.
        self.command(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = MovementController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.command(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()