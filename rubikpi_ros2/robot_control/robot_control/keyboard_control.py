
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import math
import numpy as np
import time

# TO DO: Implement this function!
def parse_waypoints(data):
    return [tuple(map(float, line.split(','))) for line in data.strip().splitlines()]

data = '''0,0,0
1,0,0
1,1,1.57
2,1,0
2,2,-1.57
1,1,-0.78
0,0,0'''
waypoints = parse_waypoints(data)

# Calibration matrix and its inverse, derived from empirical testing
CALIBRATION_MATRIX = np.array([
    [0.6327, 0.8217],
    [-4.343, 4.067],
], dtype=float)
CALIBRATION_INVERSE = np.linalg.inv(CALIBRATION_MATRIX)

MAX_MOTOR_COMMAND = 0.5 # Cannot exceed this value

# Target physical velocities used for timed movements.
# Both produce motor commands outside the measured 0.10 dead zone.
DRIVE_SPEED = 0.30      # the linear velocity how fsat we want the robot to turn
TURN_SPEED = 3.5      # angular velocity how fast we awnt to robot to turn 

class KeyboardControllerNode(Node):
    def __init__(self):
        super().__init__('keyboard_controller_node')
        self.publisher = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.current_L = 0.0
        self.current_R = 0.0
        self.running = True
        self.segment_index = 0
        self.t = 0.0
        self.dt = 0.10
        self.T = 5.0  # Hardcoded for first straight segment
        self.calibration_fault = False
        self.log_count = 0

        # Simple movement controller state
        self.actions = self.build_simple_actions()
        self.action_index = 0
        self.action_t = 0.0

        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info('Waypoint Controller Node started (SIMPLE MOVEMENT)')

    def update(self):
        # self.control_step_quintic()  # Quintic controller
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

    # === QUINTIC POLYNOMIAL CONTROLLER ===

    def quintic_coefficients(self, p0, pf, d0, df):
        T = 1.0
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3],
        ], dtype=float)
        b = np.array([p0, d0, 0, pf, df, 0], dtype=float)
        return np.linalg.solve(A, b)

    # Setting up x(t) and y(t) quintic polynomials for the current segment (plus their higher order derivatives)
    def get_velocity_acceleration_equations(self, x0, xf, y0, yf, r0, rf, d=1.0):
        distance = math.hypot(xf - x0, yf - y0)
        d = 0.5 * distance  # Geometric tangent
        self.x_t = self.quintic_coefficients(x0, xf, d * math.cos(r0), d * math.cos(rf))
        self.y_t = self.quintic_coefficients(y0, yf, d * math.sin(r0), d * math.sin(rf))
        self.vx_t = np.polynomial.polynomial.polyder(self.x_t)
        self.vy_t = np.polynomial.polynomial.polyder(self.y_t)
        self.ax_t = np.polynomial.polynomial.polyder(self.vx_t)
        self.ay_t = np.polynomial.polynomial.polyder(self.vy_t)

    def eval_vel_acc_at_t(self):
        u = min(self.t / self.T, 1.0)
        vx = np.polynomial.polynomial.polyval(u, self.vx_t) / self.T
        vy = np.polynomial.polynomial.polyval(u, self.vy_t) / self.T
        ax = np.polynomial.polynomial.polyval(u, self.ax_t) / self.T**2
        ay = np.polynomial.polynomial.polyval(u, self.ay_t) / self.T**2
        return vx, vy, ax, ay, np.hypot(vx, vy)

    def get_angular_velocity(self, vx, vy, ax, ay):
        speed_sq = vx**2 + vy**2
        if speed_sq > 1e-8:
            return (vx * ay - vy * ax) / speed_sq
        return 0.0

    def control_step_quintic(self):
            if self.calibration_fault or self.segment_index >= len(waypoints) - 1:
                self.current_L = 0.0
                self.current_R = 0.0
                return
    
            x0, y0, r0 = waypoints[self.segment_index]
            xf, yf, rf = waypoints[self.segment_index + 1]
            if self.t == 0.0:
                # Slow down on the sharper curves
                self.T = 5.0 if self.segment_index == 0 else 8.0
                self.get_velocity_acceleration_equations(x0, xf, y0, yf, r0, rf)
                self.get_logger().info(f'Segment {self.segment_index}, T={self.T:.1f}s: ({x0},{y0},{r0}) -> ({xf},{yf},{rf})')
    
            vx, vy, ax, ay, v = self.eval_vel_acc_at_t()
            omega = self.get_angular_velocity(vx, vy, ax, ay)
            self.get_left_right_speed(v, omega)
            if self.calibration_fault:
                return
    
            self.log_count += 1
            if self.log_count % 2 == 0:
                self.get_logger().info(
                    f't={self.t:.2f} segment={self.segment_index} v={v:.4f} '
                    f'omega={omega:.4f} L={self.current_L:.4f} R={self.current_R:.4f}'
                )
            self.t += self.dt
            if self.t > self.T:
                self.segment_index += 1
                self.t = 0.0

    # === SIMPLE MOVEMENT CONTROLLER ===

    def build_simple_actions(self):
        actions = []

        def wrap_angle(angle):
            return (angle + math.pi) % (2 * math.pi) - math.pi

        for i in range(len(waypoints) - 1):
            x0, y0, r0 = waypoints[i]
            xf, yf, rf = waypoints[i + 1]

            dx = xf - x0
            dy = yf - y0
            distance = math.hypot(dx, dy)

            target_heading = math.atan2(dy, dx) if distance > 1e-6 else r0

            # Initial rotation
            initial_turn = wrap_angle(target_heading - r0)

            if abs(initial_turn) > 0.02:
                omega = math.copysign(TURN_SPEED, initial_turn)
                actions.append((
                    0.0, omega,
                    abs(initial_turn) / TURN_SPEED,
                    f'Segment {i}: Initial turn'
                ))

            # Drive straight
            if distance > 1e-6:
                actions.append((
                    DRIVE_SPEED, 0.0,
                    distance / DRIVE_SPEED,
                    f'Segment {i}: Drive straight'
                ))

            # Final rotation
            final_turn = wrap_angle(rf - target_heading)

            if abs(final_turn) > 0.02:
                omega = math.copysign(TURN_SPEED, final_turn)
                actions.append((
                    0.0, omega,
                    abs(final_turn) / TURN_SPEED,
                    f'Segment {i}: Final turn'
                ))

        return actions

    def control_step_simple(self):
        if self.calibration_fault or self.action_index >= len(self.actions):
            self.current_L = 0.0
            self.current_R = 0.0
            return

        v, omega, duration, description = self.actions[self.action_index]

        if self.action_t == 0.0:
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

        if self.calibration_fault:
            return

        self.log_count += 1
        if self.log_count % 20 == 0:
            self.get_logger().info(
                f'{description}: t={self.action_t:.2f}, '
                f'L={self.current_L:.4f}, R={self.current_R:.4f}'
            )

        self.action_t += self.dt

        if self.action_t >= duration:
            time.sleep(1)  # Small delay to ensure the last command is sent
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
