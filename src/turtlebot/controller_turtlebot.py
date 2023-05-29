#!/usr/bin/env python

import rospy
import sys
import tty
import termios
import select
from std_msgs.msg import Float64MultiArray

class TurtlebotController:
    def __init__(self):
        # Create a publisher to publish wheel velocities
        self.pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray, queue_size=10)

        # Define the mapping between keys and initial wheel velocities
        self.initial_velocities = {
            'w': [0.1, 0.1],   # Forward motion
            's': [-0.1, -0.1],  # Backward motion
            'a': [-0.1, 0.1],   # Turn left
            'd': [0.1, -0.1],   # Turn right
            ' ': [0.0, 0.0]    # Stop
        }

        # Dictionary to store the current wheel velocities
        self.current_velocities = {
            'w': [0.0, 0.0],   # Forward motion
            's': [0.0, 0.0],   # Backward motion
            'a': [0.0, 0.0],   # Turn left
            'd': [0.0, 0.0],   # Turn right
            ' ': [0.0, 0.0]    # Stop
        }

    def publish_wheel_velocities(self, velocities):
        msg = Float64MultiArray()
        msg.data = velocities
        self.pub.publish(msg)

    def read_keyboard(self):
        # Save the terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())

            while not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key in self.initial_velocities:
                        if key == ' ':
                            # Reset all velocities to zero
                            for vel_key in self.current_velocities:
                                self.current_velocities[vel_key][0] = 0.0
                                self.current_velocities[vel_key][1] = 0.0
                        else:
                            self.current_velocities[key][0] += self.initial_velocities[key][0]
                            self.current_velocities[key][1] += self.initial_velocities[key][1]
                        self.publish_wheel_velocities(self.current_velocities[key])
                    elif ord(key) == 27:  # Check for Escape key
                        break

        finally:
            # Restore the terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main():
    rospy.init_node('turtlebot_controller')
    controller = TurtlebotController()
    controller.read_keyboard()


if __name__ == '__main__':
    main()