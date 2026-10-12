import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from pynput.keyboard import Key, Listener
import threading


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
        self.start_keyboard_listener()
        
        # setup timer to publish commands at 20Hz
        self.timer = self.create_timer(0.05, self.publish_motor_commands)  # 20Hz
        
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
    
    def on_press(self, key):
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
            self.get_logger().error(f'Error in on_press: {str(e)}')
    
    def on_release(self, key):
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
            self.get_logger().error(f'Error in on_release: {str(e)}')
    
    def start_keyboard_listener(self):
        """Start keyboard listener in a separate thread"""
        def listener_thread():
            with Listener(on_press=self.on_press, on_release=self.on_release, suppress=True) as listener:
                listener.join()
        
        self.listener_thread = threading.Thread(target=listener_thread, daemon=True)
        self.listener_thread.start()
        self.get_logger().info('Keyboard listener started')
    
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