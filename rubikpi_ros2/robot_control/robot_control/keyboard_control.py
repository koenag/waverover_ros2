import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
#from pynput.keyboard import Key, Listener
import threading
import math
import numpy as np


def parse_waypoints(data):
    return [
        tuple(map(float, line.split(",")))
        for line in data.strip().splitlines()
    ]


data = """0,0,0
1,0,0
1,1,1.57
2,1,0
2,2,-1.57
1,1,-0.78
0,0,0"""

waypoints = parse_waypoints(data)
print(waypoints)

class KeyboardControllerNode(Node):
    def __init__(self):
        super().__init__('keyboard_controller_node')
        
        # create publisher for motor commands
        self.publisher = self.create_publisher(
            Float32MultiArray,
            'motor_commands',
            10
        )
        
        # keyboard state
        self.pressed_keys = set()
        self.current_L = 0.0
        self.current_R = 0.0
        self.running = True
        
        # start keyboard listener
        #self.start_keyboard_listener()

        # Time/trajectory state
        self.segment_index = 0
        self.t = 0.0
        self.dt = 0.05
        self.T = 10.0
        
        
        # setup timer to publish commands at 20Hz
        self.timer = self.create_timer(0.05, self.publish_motor_commands)  # 20Hz

        # Run the control algorithm every 0.05 seconds
        self.control_timer = self.create_timer(0.05, self.control_step)
        
        self.get_logger().info('Keyboard Controller Node started')
        self.print_instructions()
        
    def print_instructions(self):
        """print control instructions"""
        instructions = """
========================================
   🚗  Robot Keyboard Controller
----------------------------------------
   W : Forward  (L=0.3, R=0.3)
   S : Backward (L=-0.3, R=-0.3)
   A : Turn Left (L=-0.5, R=0.5)
   D : Turn Right (L=0.5, R=-0.5)
   X : Stop (L=0.0, R=0.0)
  ESC: Quit
========================================
Publishing motor commands on topic: motor_commands
"""
        self.get_logger().info(instructions)



    def quintic_coefficients(self, x0, xf, T):
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ], dtype=float)

        b = np.array([x0, 0, 0, xf, 0, 0], dtype=float)

        return np.linalg.solve(A, b)
        
    
    
    def control_step(self):
        if self.segment_index >= len(waypoints) - 1:
            self.current_L = 0.0
            self.current_R = 0.0
            return

        x0, y0, r0 = waypoints[self.segment_index]
        xf, yf, rf = waypoints[self.segment_index + 1]

        # Generate coefficients once when a segment begins
        if self.t == 0.0:
            self.x_t = self.quintic_coefficients(x0, xf, self.T)
            self.y_t = self.quintic_coefficients(y0, yf, self.T)

            self.vx_t = np.polynomial.polynomial.polyder(self.x_t)
            self.vy_t = np.polynomial.polynomial.polyder(self.y_t)
            self.ax_t = np.polynomial.polynomial.polyder(self.vx_t)
            self.ay_t = np.polynomial.polynomial.polyder(self.vy_t)

        vx = np.polynomial.polynomial.polyval(self.t, self.vx_t)
        vy = np.polynomial.polynomial.polyval(self.t, self.vy_t)
        ax = np.polynomial.polynomial.polyval(self.t, self.ax_t)
        ay = np.polynomial.polynomial.polyval(self.t, self.ay_t)

        speed_sq = vx**2 + vy**2

        if speed_sq > 1e-8:
            omega = (vx * ay - vy * ax) / speed_sq
        else:
            omega = 0.0

        v = np.hypot(vx, vy)
        b = 0.127

        self.current_L = v - b * omega / 2
        self.current_R = v + b * omega / 2

        self.t += self.dt

        if self.t > self.T:
            self.segment_index += 1
            self.t = 0.0
    


        

    '''def on_press(self, key):

        """press key callback"""
        try:
            if hasattr(key, "char") and key.char:
                c = key.char.lower()
                self.pressed_keys.add(c)
                
                if c == 'w':
                    self.current_L = 0.3
                    self.current_R = 0.3
                elif c == 's':
                    self.current_L = -0.3
                    self.current_R = -0.3
                elif c == 'a':
                    self.current_L = -0.5
                    self.current_R = 0.5
                elif c == 'd':
                    self.current_L = 0.5
                    self.current_R = -0.5
                elif c == 'x':
                    self.current_L = 0.0
                    self.current_R = 0.0
                    self.pressed_keys.clear()
                    
        except Exception as e:
            self.get_logger().error(f'Error in on_press: {str(e)}')'''
    
    '''def on_release(self, key):
        """release key callback"""
        try:
            if key == Key.esc:
                self.get_logger().info('ESC pressed, shutting down...')
                self.running = False
                rclpy.shutdown()
                return False 
            
            if hasattr(key, "char") and key.char:
                c = key.char.lower()
                self.pressed_keys.discard(c)
                
                # stop if no direction keys are pressed
                if not any(k in self.pressed_keys for k in 'wasd'):
                    self.current_L = 0.0
                    self.current_R = 0.0
                    self.get_logger().debug('All direction keys released, stopping')
                    
        except Exception as e:
            self.get_logger().error(f'Error in on_release: {str(e)}')'''
    
    '''def start_keyboard_listener(self):
        """Start keyboard listener in a separate thread"""
        def listener_thread():
            with Listener(on_press=self.on_press, on_release=self.on_release, suppress=True) as listener:
                listener.join()
        
        self.listener_thread = threading.Thread(target=listener_thread, daemon=True)
        self.listener_thread.start()
        self.get_logger().info('Keyboard listener started')'''
    
    def publish_motor_commands(self):
        """Publish motor commands to ROS topic"""
        if self.running and rclpy.ok():
            try:
                msg = Float32MultiArray()
                msg.data = [self.current_L, self.current_R]
                self.publisher.publish(msg)
                
                # Log only when commands change to reduce console spam
                if hasattr(self, '_last_L') and hasattr(self, '_last_R'):
                    if self._last_L != self.current_L or self._last_R != self.current_R:
                        self.get_logger().info(f'Motor command: L={self.current_L:.2f}, R={self.current_R:.2f}')
                else:
                    self.get_logger().info(f'Motor command: L={self.current_L:.2f}, R={self.current_R:.2f}')
                
                self._last_L = self.current_L
                self._last_R = self.current_R
            except Exception as e:
                # Context may be shutting down, ignore errors
                if self.running:
                    self.get_logger().error(f'Error publishing command: {str(e)}')
    
    def destroy_node(self):
        """Cleanup operations when node is destroyed"""
        self.running = False
        
        # Send stop command before shutdown
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
        # If timer already destroyed the node, skip
        if not getattr(node, "_already_destroyed", False):
            try:
                node.destroy_node()
            except Exception:
                pass

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
        return 0  # <-- ensure exit code 0



if __name__ == '__main__':
    main()