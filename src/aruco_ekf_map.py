#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Basic imports
import numpy as np
import math
import roslib
import rospy
import tf
# from tf.broadcaster import _
from tf.transformations import quaternion_from_euler, euler_from_quaternion

# ROS messages
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, NavSatFix
from std_msgs.msg import Header, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray



def wrap_angle(ang):
    if isinstance(ang, np.ndarray):
        ang[ang > np.pi] -= 2 * np.pi
        return ang
    else:
        return ang + (2.0 * math.pi * math.floor((math.pi - ang) / (2.0 * math.pi)))


class Robot:
    """
    Class that represents a Robot.
    """

    def __init__(self):
        # Robot physical parameters
        self.b = 0.23
        self.r = 0.035

        # Wheel velocities
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0

        # Linear and angular speeds
        self.v = 0
        self.w = 0

        # Flags and time initialization
        self.left_wheel_received = False
        self.last_time = rospy.Time.now()

        # Robot position initialization
        # self.x = 0.0
        # self.y = 0.0
        # self.th = 0
        self.x = 3.0
        self.y = -0.78 
        self.th = np.pi/2

        # Robot pose initialization
        self.xk = np.array([[self.x], [self.y], [self.th]])
        self.Pk = 0.2 * np.eye(3)

        # Observation initialization
        self.z = np.array([[0.0], [0.0]])

        # Innovation initialization
        self.v_ino = np.array([[0.0], [0.0]])

        # Map initialization
        # professor map
        self.aruco_id_map = {
            '0': [0.3, 0.0, -0.1],
            '1': [1.4, -0.2, -0.15],
            '11': [3.99, 0.605, -0.15],
            '21': [2.715, 2.5, -0.15],
            '31': [2.535, 2.39, -0.15],
            '41': [2.1, 1.2, -0.15],
            '51': [1.99, 1.2, -0.15],
            '61': [4.57, -1.69, -0.15],
            '71': [0.283, 1.70, -0.15],
            '81': [0.80, 2.86, -0.15]
        }

        # Model noise initialization
        self.right_wheel_noise_sigma = 0.2
        self.left_wheel_noise_sigma = 0.2
        self.Qk = np.array([[self.right_wheel_noise_sigma**2, 0],
                            [0, self.left_wheel_noise_sigma**2]])

        
        self.sub = rospy.Subscriber(
            '/turtlebot/joint_states', JointState, self.joint_state_callback)
        
        self.odom_pub = rospy.Publisher(
            '/turtlebot/odom', Odometry, queue_size=10)
        
        self.modem_sub = rospy.Subscriber(
            'measurd_data', Float32MultiArray, self.aruco_detection)
        
        self.floatmat_pub = rospy.Publisher("Mydata_xyth",Float32MultiArray,queue_size=10)
        self.floatmat1_pub = rospy.Publisher("Mydata_pk",Float32MultiArray,queue_size=10)
        self.floatmat2_pub = rospy.Publisher("Mydata_id_sensor_obzer",Float32MultiArray,queue_size=10)
        
        self.tf_br = tf.TransformBroadcaster()

    def joint_state_callback(self, msg):
        if msg.name[0] == "turtlebot/kobuki/wheel_left_joint":
            self.left_wheel_vel = msg.velocity[0]
            self.left_wheel_rec = True

        elif msg.name[0] == "turtlebot/kobuki/wheel_right_joint":
            self.right_wheel_vel = msg.velocity[0]

            if self.left_wheel_rec:
                # calculation
                left_lin_vel = self.left_wheel_vel * self.r
                right_lin_vel = self.right_wheel_vel * self.r

                self.v = (left_lin_vel + right_lin_vel) / 2.0
                self.w = (left_lin_vel - right_lin_vel) / self.b

                # print("w ", self.w)

                # calculate dt
                current_time = rospy.Time.from_sec(
                    msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
                dt = (current_time - self.last_time).to_sec()
                # print("dt", dt)
                self.last_time = current_time

                # reset flag
                self.left_wheel_rec = False

            
                self.floatmat_pub.publish(Float32MultiArray(data=self.xk))

                uncertnity = self.Pk.copy()
                uncertnity = uncertnity.flatten()
                

                self.floatmat1_pub.publish(Float32MultiArray(data = uncertnity))

                # Prediction step
                self.prediction(dt)
                # print("xk",self.xk)

                # Odom path publisher
                self.odom_path_pub()

    def aruco_detection(self, msg):
        
        # Update the observation model with data from the message
        self.z[0] = msg.data[1]
        self.z[1] = wrap_angle(msg.data[2])
        print("Aruco Id", msg.data[0])
        print("z", self.z)

        # Calculate the change in position based on the Aruco ID map
        delta_x = float(-(self.xk[0]) +
                        self.aruco_id_map[str(int(msg.data[0]))][0])
        delta_y = float(-(self.xk[1]) +
                        self.aruco_id_map[str(int(msg.data[0]))][1])
        
        r = math.sqrt((delta_x)**2 + (delta_y)**2)
        a = float(wrap_angle(math.atan2((delta_y), (delta_x)) - self.xk[2]))
        
        hxk = np.array([[r],
                        [a]])
        print("hxk", hxk)

        all_measured = [msg.data[0], self.z[0], self.z[1], r, a ]

        self.floatmat2_pub.publish(Float32MultiArray(data = all_measured))
       
        # Calculate the measurement jacobian
        H = np.array([[-delta_x / math.sqrt(delta_x**2 + delta_y**2),
                       -delta_y / math.sqrt(delta_x**2 + delta_y**2),
                       0],
                      [delta_y / (delta_x**2 + delta_y**2),
                        delta_x / (delta_x**2 + delta_y**2),
                        -1]])

        # Add sensor noise to the measurement covariance matrix
        range_sigma = 0.2
        azmith_sigma = 0.2
        Rk = np.array([[range_sigma**2, 0],
                       [0, azmith_sigma**2]])

        # Update the state estimate based on the observation
        self.update(hxk, H, Rk)

        # # Odom path publisher
        self.odom_path_pub()

    def prediction(self, dt):
        """
        Predicts the next state of the robot using the motion model.

        Args:
        - dt: The time interval between the current state and the next state.

        Returns:
        - None

        """

        # Calculate Jacobians with respect to state vector
        H = np.array([[1, 0, -math.sin(float(self.xk[2]))*(self.v)*dt],
                      [0, 1, math.cos(float(self.xk[2]))*(self.v)*dt],
                      [0, 0, 1]])

        # Calculate Jacobians with respect to noise
        Wk = np.array([[0.5 * math.cos(float(self.xk[2]))*dt, 0.5 * math.cos(float(self.xk[2]))*dt],
                       [0.5 * np.sin(float(self.xk[2]))*dt, 0.5 *
                        math.sin(float(self.xk[2]))*dt],
                       [-dt/self.b, dt/self.b]])

        # Update the prediction "uncertainty"
        self.Pk = H @ self.Pk @ np.transpose(H) + \
            Wk @ self.Qk @ np.transpose(Wk)

        # Integrate position
        self.xk[0] = self.xk[0] + math.cos(float(self.xk[2]))*(self.v)*dt
        self.xk[1] = self.xk[1] + math.sin(float(self.xk[2]))*(self.v)*dt
        self.xk[2] = wrap_angle(self.xk[2] + (self.w)*dt)

    def update(self, hxk, H, Rk):
        """
        Update the state of the Kalman Filter with the latest measurement.

        Parameters:
            hxk (numpy array): Observation model output.
            H (numpy array): Observation model Jacobian.
            Rk (numpy array): Sensor noise covariance matrix.

        Returns:
            None.
        """

        # Compute the innovation vector.
        self.v_ino[0] = self.z[0] - hxk[0]
        self.v_ino[1] = wrap_angle(self.z[1] - hxk[1])
        print("zk",self.z)
        print("self.v_ino",  self.v_ino)

        # Compute the innovation covariance.
        S = H @ self.Pk @ np.transpose(H) + Rk

        # Compute the Kalman gain.
        I = np.eye(3)
        K = self.Pk @ np.transpose(H) @ np.linalg.inv(S)

        # Update the state estimate.
        self.xk = self.xk + K @ self.v_ino
        self.xk[2] = wrap_angle(self.xk[2])

        # Update the state covariance.
        self.Pk = (I - K @ H) @ self.Pk @ np.transpose(I -
                                                       K @ H) + K @ Rk @ np.transpose(K)

    def odom_path_pub(self):

        # Transform theta from euler to quaternion
        q = quaternion_from_euler(0, 0, float(wrap_angle(self.xk[2])))

        # Publish predicted odom
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world_ned"
        odom.child_frame_id = "turtlebot/kobuki/base_footprint"

        odom.pose.pose.position.x = self.xk[0]
        odom.pose.pose.position.y = self.xk[1]

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.pose.covariance = [self.Pk[0, 0], self.Pk[0, 1], 0, 0, 0, self.Pk[0, 2],
                                self.Pk[1, 0], self.Pk[1, 1], 0, 0, 0, self.Pk[1, 2],
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                self.Pk[2, 0], self.Pk[2, 1], 0, 0, 0, self.Pk[2, 2]]

        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w

        self.odom_pub.publish(odom)

        self.tf_br.sendTransform((float(self.xk[0]), float(self.xk[1]), 0.0), q, rospy.Time.now(
        ), odom.child_frame_id, odom.header.frame_id)


if __name__ == '__main__':

    rospy.init_node('aruco_ekf_map')

    robot = Robot()

    rospy.spin()

# ===============================================================================


##################################################################################################
