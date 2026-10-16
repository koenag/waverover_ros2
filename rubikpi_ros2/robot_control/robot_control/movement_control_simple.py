
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import math
import numpy as np

# Calibration matrix and its inverse, derived from empirical testing
CALIBRATION_MATRIX = np.array([
    [0.565034, 0.733764],
    [-4.08242, 3.82298],
], dtype=float)

CALIBRATION_INVERSE = np.linalg.inv(CALIBRATION_MATRIX)

MAX_MOTOR_COMMAND = 0.5 # L,R cannot exceed this value

# Pre-determined drive speed (linear velocity) and turn speed (angular velocity)
DRIVE_SPEED = 0.30      # the linear velocity how fast we want the robot to turn
TURN_SPEED = 3.0      # angular velocity how fast we want to robot to turn 

def parse_waypoints(file):
    with open(file, 'r') as f:
        data = f.read()
    return [tuple(map(float, line.split(','))) for line in data.strip().splitlines()]

waypoints = parse_waypoints("./waypoint.txt") 

class KeyboardControllerNode(Node):
    def __init__(self):
        super().__init__('keyboard_controller_node')
        self.publisher = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.current_L = 0.0
        self.current_R = 0.0
        self.running = True

        self.segment_index = 0
        self.dt = 0.10
    
        self.calibration_fault = False
        self.log_count = 0

        # Simple movement controller state
        self.actions = self.build_simple_actions() # list of [(linear velocity, angular velocity, time, description)]
        self.action_index = 0
        self.action_t = 0.0

        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info('Waypoint Controller Node started (SIMPLE MOVEMENT)')

    def update(self):
        self.control_step_simple()      # Simple movement controller
        self.publish_motor_commands()

    # Compute left and right motor speeds from desired linear/angular velocities using the calibration matrix.
    def get_left_right_speed(self, v, omega):
            desired_velocity = np.array([v, omega], dtype=float)
            L, R = CALIBRATION_INVERSE @ desired_velocity
            # Error if the computed motor commands are not finite or exceed the maximum allowed command
            if not np.all(np.isfinite([L, R])) or max(abs(L), abs(R)) > MAX_MOTOR_COMMAND:
                self.current_L = 0.0
                self.current_R = 0.0
                self.calibration_fault = True
                self.get_logger().error(
                    f'Unachievable command: v={v:.4f}, omega={omega:.4f}, L={L:.4f}, R={R:.4f}. Stopping.'
                )
                return
            self.current_L = float(L)
            self.current_R = float(R)

    # Turtle graphics get the actions with linear velocity, angular, and time required to execute step 
    def build_simple_actions(self):
        actions = []

        # keep angle between -pi and pi (i.e 3pi/2 turn maps to -pi/2)
        def wrap_angle(angle):
            wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
            if abs(wrapped + math.pi) < 0.01:
                return math.pi
            return wrapped

        for i in range(len(waypoints) - 1):
            # r0 is intial rotation, rf is final rotation
            x0, y0, r0 = waypoints[i] 
            xf, yf, rf = waypoints[i + 1]

            dx = xf - x0
            dy = yf - y0
            distance = math.hypot(dx, dy)

            angle_to_next_point = math.atan2(dy, dx) if distance > 1e-6 else r0

            # How much to initially rotate to get to next point
            initial_turn = wrap_angle(angle_to_next_point - r0)

            t_initial = abs(initial_turn) / TURN_SPEED
            if abs(initial_turn) > 0.02: 
                omega = math.copysign(TURN_SPEED, initial_turn)
                actions.append((
                    0.0, omega,
                    t_initial,
                    f'Segment {i}: Initial turn'
                ))
                actions.append((0.0, 0.0, 0.4, 'Pause')) # Temporary pause for movement distinction

            t_drive = distance / DRIVE_SPEED
            # Drive straight 
            if distance > 1e-6: 
                actions.append((
                    DRIVE_SPEED, 0.0,
                    t_drive,
                    f'Segment {i}: Drive straight'
                ))

            
            # How much to rotate at the end to face desired direction
            final_turn = wrap_angle(rf - angle_to_next_point)

            t_final  = abs(final_turn) / TURN_SPEED
            if abs(final_turn) > 0.02:
                omega = math.copysign(TURN_SPEED, final_turn)
                actions.append((
                    0.0, omega,
                    t_final,
                    f'Segment {i}: Final turn'
                ))
                actions.append((0.0, 0.0, 0.4, 'Pause')) # Temporary pause for movement distinction

        return actions

    # Decide current_L, current_R
    def control_step_simple(self):
        if self.calibration_fault or self.action_index >= len(self.actions):
            self.current_L = 0.0
            self.current_R = 0.0
            return

        v, omega, duration, description = self.actions[self.action_index]

        # Get waypoints at beginning of segment (PURELY FOR LOGGING)
        if self.action_t == 0.0 and description != 'Pause':
            # Identify the waypoint pair
            segment_number = int(description.split(':')[0].split()[1])

            start = waypoints[segment_number]
            end = waypoints[segment_number + 1]

            self.get_logger().info(
                f'\n===== SEGMENT {segment_number} =====\n'
                f'FROM waypoint {segment_number}: {start}\n'
                f'TO waypoint {segment_number + 1}: {end}\n'
                f'Action: {description}\n'
                f'Duration: {duration:.3f}s'
            )

            if abs(omega) > 0.001:
                direction = 'LEFT (CCW)' if omega > 0 else 'RIGHT (CW)'
                angle_deg = math.degrees(omega * duration)

                self.get_logger().info(
                    f'ROTATION START: {direction}, '
                    f'rotating at angle={angle_deg:.1f} degrees, '
                    f'omega={omega:.3f} rad/s'
                )

        self.get_left_right_speed(v, omega)

        # if current_L, current_R oob
        if self.calibration_fault:
            return

        self.log_count += 1
        if self.log_count % 2 == 0:
            self.get_logger().info(
                f'{description}: t={self.action_t:.2f}, '
                f'L={self.current_L:.4f}, R={self.current_R:.4f}'
            )

        self.action_t += self.dt

        if self.action_t >= duration:
            self.get_logger().info(
                f'ACTION COMPLETED: {description}'
            )

            self.action_index += 1
            self.action_t = 0.0

            if self.action_index >= len(self.actions):
                self.current_L = 0.0
                self.current_R = 0.0
                self.get_logger().info('ALL WAYPOINTS COMPLETED')

    # Publish the current left and right motor commands to the 'motor_commands' topic.
    def publish_motor_commands(self):
        if self.running and rclpy.ok():
            msg = Float32MultiArray()
            msg.data = [self.current_L, self.current_R]
            self.publisher.publish(msg)

    def destroy_node(self):
        self.running = False
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0]
        self.publisher.publish(msg)
        self.get_logger().info('Stop command sent')
        super().destroy_node()



def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
