import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32MultiArray


# Motion settings
INTRINSTIC_CALIBRATION_MATRIIX = np.array([
    [0.565034, 0.733764],
    [-4.08242, 3.82298]
])
MAX_MOTOR_COMMAND = 0.5
DRIVE_SPEED = 0.2
TURN_SPEED = 3.0
POSITION_TOLERANCE = 0.05
HEADING_TOLERANCE = 0.2
MAX_DRIVE_SEGMENT = 0.20

# Localization search settings
SEARCH_INITIAL_PAUSE = 2.0
SEARCH_ROTATION_TIME = 0.10
SEARCH_PAUSE = 3.0

CALLBACK_FREQUENCY = 0.05
REPORT_PERIOD = 0.5

#robot settings:
ROBOT_RADIUS = 0.10

OFFSET = (0.7, 0.4, 0)


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class MovementController(Node):
    def __init__(self):
        '''
        Publishes L,R to motor_commands topic
        Subscribes to robot_pose 
        '''
    
        super().__init__('movement_controller_node')
        

        self.publisher = self.create_publisher(
            Float32MultiArray, 'motor_commands', 10)
        
        self.create_subscription(
            Pose2D, '/robot_pose', self.update_pose, 10)
       
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
        self.pose = None
        self.pose_time = 0.0
        self.used_pose_time = 0.0
        self.total_linear_distance_travelled = 0
        self.total_angular_distance_travelled  = 0

        self.pause_until = None
      

        # (linear speed, angular speed, end time, movement_description)
        self.movement = None

        # (search phase, phase start time)
        self.search = None

        self.create_timer(CALLBACK_FREQUENCY, self.move)

        # 2D Trajectory history
        self.position_history = [OFFSET]
        self.position_estimate= OFFSET
        self.create_timer(REPORT_PERIOD, self.report_position_history)

        # LOG: Controller initialization
        self.get_logger().info(
            f'[START] {len(self.waypoints)} waypoints loaded | '
            f'Drive speed={DRIVE_SPEED:.2f}m/s | '
            f'Turn speed={TURN_SPEED:.2f}rad/s'
        )

    def update_pose(self, message):
        '''
        Find what the pose it as save it to self.pose (as a class variable so we can access it when we adjust robot movement)
        '''
        self.pose = (message.x, message.y, message.theta)
        self.pose_time = time.monotonic()

    def report_position_history(self):
        if self.movement:
            self.position_history.append(self.position_estimate)
        elif self.pose is not None:
            self.position_history.append(self.pose)
        else:
            self.position_history.append(self.position_history[-1] if self.position_history else self.position_estimate)

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

    def create_movement(self, linear_speed, angular_speed, amount,starting_pose, movement_description):
        actual_linear, actual_angular = self.command(
            linear_speed, angular_speed)

        actual_speed = abs(
            actual_linear if linear_speed != 0 else actual_angular)

        duration = amount / max(actual_speed, 1e-6)
        movement_start_time = time.monotonic()
        self.movement = (
            linear_speed, angular_speed, actual_angular, movement_start_time, duration, starting_pose, movement_description)

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
        Perform bursts of 'open-loop' movement before checking position and heading with an April Tag in view
        '''
        
        now = time.monotonic()
        if self.pause_until and now < self.pause_until:
            self.command(0.0, 0.0)
            return

        # perform a burst of movement, don't respond to an April Tag in this time
        # if reached the end of the movement (based on time) stop movement and record the last post time used
        # otherwise enforce the speeds
        if self.movement:
            linear_speed, angular_speed, angular_rotation, movement_start_time, duration, starting_pose, movement_description = self.movement
            
            if now < movement_start_time + duration:
                self.command(linear_speed, angular_speed)
                return
            
            expected_linear_distance_travelled = linear_speed * duration
            expected_angular_distance_travelled = angular_rotation * ROBOT_RADIUS
            self.total_linear_distance_travelled += expected_linear_distance_travelled 
            self.total_angular_distance_travelled  += expected_angular_distance_travelled
            
            # Update position estimate based on the last known position and the expected distance travelled
            self.position_estimate = (starting_pose[0] + expected_linear_distance_travelled * math.cos(self.position_history[-1][2]),
                                       starting_pose[1] + expected_linear_distance_travelled * math.sin(self.position_history[-1][2]),
                                       starting_pose[2] + angular_speed * duration)
            
        

            self.command(0.0, 0.0)
            self.movement = None
            self.used_pose_time = self.pose_time
            self.search = None

            # LOG: Movement completed
            self.get_logger().info(
                f'[MOVE END] {movement_description} completed | '
                f'Waiting for a new AprilTag pose'
            )
            return

        # All waypoints completed.
        if self.waypoint_index >= len(self.waypoints):
            self.command(0.0, 0.0)
            self.get_logger().info(f"LINEAR DISTANCE TRAVELLED: {self.total_linear_distance_travelled}")
            self.get_logger().info(f"ANGULAR DISTANCE TRAVELLED: {self.total_angular_distance_travelled}")
            self.get_logger().info(f"2D TRAJECTORY HISTORY: {self.position_history}")
            
            return

        # once you're done with the movement check for repositioning if you can see an april tag / have gotten a new one

        # Handles case where you knew where you were at a previous time step and you got a new april tag input (ie you can determine your new location)
        if self.pose is not None and self.pose_time > self.used_pose_time:
            self.used_pose_time = self.pose_time
            self.search = None

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
                # reached the end of segment and either need to turn that final rotation (in which case turn)
                # or are within heading tolerance in which case mark this segment as done
                if abs(final_heading_error) <= HEADING_TOLERANCE:

                    # LOG: Waypoint successfully completed
                    self.get_logger().info(
                        f'[WAYPOINT REACHED] {self.waypoint_index} | '
                        f'Position error={distance_remaining:.3f}m | '
                        f'Heading error={math.degrees(final_heading_error):+.1f}deg'
                    )
                    self.pause_until = time.monotonic() + 1.0
                    self.waypoint_index += 1

                    if self.waypoint_index < len(self.waypoints):
                       
                        self.get_logger().info(
                            f'[NEXT WAYPOINT] Moving to WP {self.waypoint_index}: '
                            f'{self.waypoints[self.waypoint_index]}'
                        )
                    else:
                        # LOG: Navigation finished
                        self.get_logger().info(
                            '[DONE] All waypoints completed'
                        )

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

                        abs(final_heading_error), self.pose,'FINAL TURN'
                    )

            # not in the right position and are facing the wrong way, in which case rotate along
            # the error to correct your heading
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
                    
                    abs(heading_error),self.pose, 'TURN'
                )

            # facing the right way, but not yet reached the end of the segment, in which case keep moving forward
            else:
                # Slow down when approaching the waypoint.
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

        # No new pose: stop while waiting for localization.
        self.command(0.0, 0.0)

        # last case, you don't see an april tag and/or don't know where you are. Solution: Rotate until you see a tag
        if self.pose is None or self.pose_time <= self.used_pose_time:
            
            # Added a brief inital pause to handle any starting delays with launch files
            if self.search is None:
                self.search = ('INITIAL_PAUSE', now)

                # LOG: Localization unavailable
                self.get_logger().warning(
                    '[SEARCH] No AprilTag pose available; starting search'
                )

            phase, start_time = self.search
            elapsed = now - start_time


            # Can either be in a pause or rotate state. After being in one state for more than MACROS tiime, switch to the other state
            if phase == 'INITIAL_PAUSE' and elapsed >= SEARCH_INITIAL_PAUSE:
                self.search = ('ROTATE', now)

                # LOG: Beginning search rotation
                self.get_logger().info(
                    '[SEARCH] Rotating LEFT/CCW to find AprilTag'
                )

            elif phase == 'ROTATE' and elapsed >= SEARCH_ROTATION_TIME:
                self.search = ('PAUSE', now)

                # LOG: Search rotation completed
                self.get_logger().info(
                    '[SEARCH] Rotation finished; waiting for AprilTag'
                )

            elif phase == 'PAUSE' and elapsed >= SEARCH_PAUSE:
                self.search = ('ROTATE', now)

                # LOG: Retrying localization search
                self.get_logger().info(
                    '[SEARCH] Still no AprilTag; rotating again'
                )

            # Do the rotation
            if self.search[0] == 'ROTATE':
                self.command(0.0, TURN_SPEED)


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
