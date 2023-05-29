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
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState, NavSatFix
from std_msgs.msg import Header, Float32MultiArray
from geometry_msgs.msg import PoseStamped, Point
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
        # self.y = -0.0
        # self.th = 0
        self.x = 3.0
        self.y = -0.78 
        self.th = np.pi/2

        # Robot pose initialization
        self.xk = np.array([[self.x], [self.y], [self.th]])
        self.current_pose = [self.xk[0], self.xk[1], self.xk[2]]
        self.Pk = 0.2*0.2*np.eye(3)

        # Observation initialization
        self.z = np.array([[0.0], [0.0]])

        # Innovation initialization
        self.v_ino = np.array([[0.0], [0.0]])

        # Map initialization

        # my map
        # self.aruco_id_map = {
        #     '0': [0.3, 0.0, -0.1],
        #     '1': [0.3, 0.15, -0.1],
        #     '2': [5.0, -5.0, -2.0],
        #     '4': [-5.0, -5.0, -2.0]
        # }
        # professor map
        # self.aruco_id_map = {
        #     '0': [0.3, 0.0, -0.1],
        #     '1': [1.4, -0.2, -0.15],
        #     '11': [3.99, 0.605, -0.15],
        #     '21': [2.715, 2.5, -0.15],
        #     '31': [2.535, 2.39, -0.15],
        #     '41': [2.01, 1.2, -0.15],
        #     '51': [1.99, 1.2, -0.15],
        #     '61': [4.57, -1.69, -0.15],
        #     '71': [0.283, 1.70, -0.15],
        #     '81': [0.80, 2.86, -0.15]
        # }
        self.arucos = []

        # Model noise initialization
        self.right_wheel_noise_sigma = 0.2
        self.left_wheel_noise_sigma = 0.2
        self.Qk = np.array([[self.right_wheel_noise_sigma**2, 0],
                            [0, self.left_wheel_noise_sigma**2]])

        # Odom publisher and subscriber initialization
        # self.sub = rospy.Subscriber(
        #     '/kobuki/joint_states', JointState, self.joint_state_callback)
        self.sub = rospy.Subscriber(
            '/turtlebot/joint_states', JointState, self.joint_state_callback)
        
        self.odom_pub = rospy.Publisher(
            '/turtlebot/odom', Odometry, queue_size=10)
        
        self.modem_sub = rospy.Subscriber(
            'measurd_data', Float32MultiArray, self.aruco_detection)
        
        self.path_pub = rospy.Publisher('/robot_path', Path, queue_size=10)


        self.odom_sub = rospy.Subscriber('/turtlebot/kobuki/ground_truth', Odometry, self.get_odom)

        self.pathgt_pub = rospy.Publisher('/gt_path', Path, queue_size=10)
        
        self.arucos_belief_pub = rospy.Publisher("/arucos_belief", MarkerArray)
        self.arucos_h_pub = rospy.Publisher("/arucos_obzerved_h", MarkerArray)
        self.arucos_z_pub = rospy.Publisher("/arucos_obzerved_z", MarkerArray)


        self.floatmatar_pub = rospy.Publisher("Mydata_xytharu",Float32MultiArray,queue_size=10)
        self.floatmat_pub = rospy.Publisher("Mydata_xyth",Float32MultiArray,queue_size=10)

        self.floatmatgt_pub = rospy.Publisher("Mydatagt_xyth",Float32MultiArray,queue_size=10)
        self.floatmat1_pub = rospy.Publisher("Mydata_pk",Float32MultiArray,queue_size=10)
        self.floatmat2_pub = rospy.Publisher("Mydata_id_sensor_obzer",Float32MultiArray,queue_size=10)
        self.floatmat3_pub = rospy.Publisher("Arucos_detected_order",Float32MultiArray,queue_size=10)

        # Create a Path message
        #odom path
        self.path_msg = Path()
        self.path_msg.header.stamp = rospy.Time.now()
        #gt path
        self.pathgt_msg = Path()
        self.pathgt_msg.header.stamp = rospy.Time.now()

        self.tf_br = tf.TransformBroadcaster()


    # Odometry callback: Gets current robot pose and stores it into self.current_pose
    def get_odom(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x,
                                                              odom.pose.pose.orientation.y,
                                                              odom.pose.pose.orientation.z,
                                                              odom.pose.pose.orientation.w])
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
        self.floatmatgt_pub.publish(Float32MultiArray(data=self.current_pose))
        # self.gt_path()
        # # odometry path
        # self.robot_path()
        

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

                self.floatmatar_pub.publish(Float32MultiArray(data=self.xk))
                self.floatmat_pub.publish(Float32MultiArray(data=[self.xk[0],self.xk[1],self.xk[2]]))

                uncertnity = self.Pk.copy()
                uncertnity = uncertnity.flatten()
                

                self.floatmat1_pub.publish(Float32MultiArray(data = uncertnity))


                # Prediction step
                self.prediction(dt)

                # Odom path publisher
                self.odom_path_pub()

                

    def aruco_detection(self, msg):
        
        """
        Performs detection of an ArUco feature and updates the SLAM system accordingly.
        
        Args:
            msg (object): Message containing ArUco feature data.
            
        Returns:
            None
        """
            
        # Update the observation model with data from the message
        self.z[0] = msg.data[1]  # Update range measurement
        self.z[1] = wrap_angle(msg.data[2])  # Update azimuth measurement
        aruco_id = msg.data[0]  # Get the ArUco feature ID
        # print("xk",self.xk)
        if aruco_id not in self.arucos:
            # New feature detected, add it to the SLAM system
            print("aruco_id", aruco_id)
            self.add_new_feature(self.z[0], self.z[1], aruco_id)
        else:
            # Existing feature detected, perform measurement update

            print("Aruco Id", aruco_id)
            
            # Add sensor noise to the measurement covariance matrix
            range_sigma = 0.5
            azimuth_sigma = 0.25
            Rk = np.array([[range_sigma**2, 0],
                        [0, azimuth_sigma**2]])

            delta_x = float(-self.xk[0] + self.xk[self.arucos.index(aruco_id) + 3])
            delta_y = float(-self.xk[1] + self.xk[self.arucos.index(aruco_id) + 4])

            H_list = np.zeros((2, len(self.arucos)))

            print("observation from sensor", self.z)
            
            r = math.sqrt(delta_x**2 + delta_y**2)
            a = float(wrap_angle(wrap_angle(math.atan2(delta_y, delta_x)) - wrap_angle(self.xk[2])))

            # Convert predicted and measured coordinates to world frame
            x_h, y_h = self.T_R_W(r, a)
            x_z, y_z = self.T_R_W(self.z[0], self.z[1])
            
            print("x_h, y_h", x_h, y_h)
            
            # Publish markers for visualization
            self.markerpub_h(x_h, y_h, aruco_id)
            self.markerpub_z(x_z, y_z, aruco_id)
            
            hxk = np.array([[r],
                            [a]])
            
            print("observation from equation hxk", hxk)

            # Calculate the Jacobian matrix H
            H = np.array([[-delta_x / math.sqrt(delta_x**2 + delta_y**2),
                        -delta_y / math.sqrt(delta_x**2 + delta_y**2),
                        0],
                        [delta_y / (delta_x**2 + delta_y**2),
                        delta_x / (delta_x**2 + delta_y**2),
                        -1]])

            H_list = np.hstack((H, H_list))
            
            J1 = np.array([[delta_x / math.sqrt(delta_x**2 + delta_y**2),
                            delta_y / math.sqrt(delta_x**2 + delta_y**2)],
                        [delta_y / (delta_x**2 + delta_y**2),
                            delta_x / (delta_x**2 + delta_y**2)]])

            print("J1", J1)
            
            N = self.arucos.index(aruco_id) + 3
            H_list[0:2, N: N+2] = J1
            
            all_measured = [aruco_id, self.z[0], self.z[1], r, a]
            self.floatmat2_pub.publish(Float32MultiArray(data=all_measured))
            self.floatmat3_pub.publish(Float32MultiArray(data=self.arucos))
            
            # Update the SLAM system using the measurement and Jacobian matrix
            self.update(hxk, H_list, Rk)

            # Publish the odom path
            self.odom_path_pub()

            # Publish markers
            self.markerpub()

            return

          

    def prediction(self, dt):
        """
        Predicts the next state of the robot using the motion model.

        Args:
        - dt: The time interval between the current state and the next state.

        Returns:
        - None

        """

        # Calculate Jacobians with respect to state vector
        Ak = np.array([[1, 0, -math.sin(float(self.xk[2]))*(self.v)*dt],
                      [0, 1, math.cos(float(self.xk[2]))*(self.v)*dt],
                      [0, 0, 1]])

        # Calculate Jacobians with respect to noise
        Wk = np.array([[0.5 * math.cos(float(self.xk[2]))*dt, 0.5 * math.cos(float(self.xk[2]))*dt],
                       [0.5 * np.sin(float(self.xk[2]))*dt, 0.5 *
                        math.sin(float(self.xk[2]))*dt],
                       [-dt/self.b, dt/self.b]])

        # Update the prediction "uncertainty" befor slam
        # self.Pk = H @ self.Pk @ np.transpose(H) + \
        #     Wk @ self.Qk @ np.transpose(Wk)

        # Integrate position
        self.xk[0] = self.xk[0] + math.cos(float(self.xk[2]))*(self.v)*dt
        self.xk[1] = self.xk[1] + math.sin(float(self.xk[2]))*(self.v)*dt
        self.xk[2] = wrap_angle(self.xk[2] + (self.w)*dt)

        n = int((self.xk.size-3)/2)
        F_k = np.eye(3 + 2*n, 3 + 2*n)
        F_2k = np.zeros((3 + 2*n, 2))

        F_k[0:3, 0:3] = Ak
        F_2k[0:3, 0:2] = Wk
        
        tmp = self.Pk
        Pk = np.zeros((3 + 2*n, 3 + 2*n))
        Pk[0:self.Pk.shape[0], 0:self.Pk.shape[1]] = tmp
        self.Pk = Pk

        # print("F_k", F_k)
        # print("F_2k", F_2k)
        self.Pk = F_k@self.Pk@np.transpose(F_k) + F_2k@self.Qk@np.transpose(F_2k)
    


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
        self.v_ino[1] = (self.z[1] - hxk[1])
        print("self.v_ino",  self.v_ino)

        # Compute the innovation covariance.
        S = H @ self.Pk @ np.transpose(H) + Rk

        # Compute the Kalman gain.
        # I = np.eye(3)
        I = np.eye(self.xk.shape[0])
        K = self.Pk @ np.transpose(H) @ np.linalg.inv(S)

        # Mahalanobis distance
        D = np.transpose(self.v_ino)@np.linalg.inv(S)@self.v_ino
        chi_thres = 0.3 # chi square 2DoF 95% confidence

        if D < chi_thres:

            print(f"will update {D}")
            # Update the state estimate.
            self.xk = self.xk + K @ self.v_ino
            # print("xk",self.xk)
            self.xk[2] = wrap_angle(self.xk[2])

            # Update the state covariance.
            self.Pk = (I - K @ H) @ self.Pk @ np.transpose(I - K @ H) + K @ Rk @ np.transpose(K)
        else:
            print("more thn chi-square will not update")

    def add_new_feature(self, range, azimuth, id):
        """
        Adds a new feature to the SLAM system based on range and azimuth measurements.
        
        Args:
            range (float): Range measurement of the new feature.
            azimuth (float): Azimuth measurement of the new feature.
            id (int): Identifier of the new feature.
            
        Returns:
            None
        """
            
        # Append the new feature ID and a placeholder value to the arucos list
        self.arucos.append(id)
        self.arucos.append(99999)

        # Convert polar coordinates of the new feature to Cartesian coordinates in the world frame
        x_predict = range * math.cos(azimuth)  # x-coordinate prediction
        y_predict = range * math.sin(azimuth)  # y-coordinate prediction

        x_wo = x_predict * math.cos(self.xk[2]) - y_predict * math.sin(self.xk[2]) + self.xk[0]  # world x-coordinate
        y_wo = x_predict * math.sin(self.xk[2]) + y_predict * math.cos(self.xk[2]) + self.xk[1]  # world y-coordinate

        print("xw, yw {}, {}".format(x_wo, y_wo))

        N = len(self.xk)

        # Add the new feature's Cartesian coordinates to the state vector
        self.xk = np.vstack((self.xk, x_wo))
        self.xk = np.vstack((self.xk, y_wo))

        # Calculate the Jacobian matrix J_1
        J_1 = np.array([[1, 0, float(-x_predict*math.sin(self.xk[2])-y_predict*math.cos(self.xk[2]))],
                        [0, 1,  float(x_predict*math.cos(self.xk[2])-y_predict*math.sin(self.xk[2]))]])

        # Calculate the Jacobian matrix J_2
        J_2 = np.array([[float(math.cos(float(self.xk[2]) + azimuth)), float(math.sin(float(self.xk[2]) + azimuth))],
                        [float(-range*(math.sin(float(self.xk[2]) + azimuth))), float(-range*math.sin(self.xk[2])*math.sin(azimuth)) + float(range*math.cos(self.xk[2])*math.cos(azimuth))]])

        # print("J2", J_2)

        F_k_ = np.eye(N)  # Create an identity matrix of size (3 + 2n) * (3 + 2n)

        # Construct the G_1K matrix by stacking F_k_ and J_1
        if N < 4:
            G_1K = np.vstack((F_k_, J_1))
        else:
            new = np.zeros((2, N))
            new[0:2, 0:3] = J_1
            G_1K = np.vstack((F_k_, new))

        G_2k = np.zeros((N + 2, 2))  # Create a matrix of zeros of size (3 + 2n + 2) * 2
        G_2k[N: N + 2, 0:2] = J_2

        # Update the covariance matrix Pk using the matrices G_1K, G_2k, and self.Qk
        self.Pk = G_1K @ self.Pk @ np.transpose(G_1K) + G_2k @ self.Qk @ np.transpose(G_2k)

        return

    
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

    def markerpub(self):
        
        ma = MarkerArray()
        for idx in range(0, len(self.arucos), 2):
            arucos_num = self.arucos[idx]  # get the ArUco marker number

            marker = Marker()
            marker.header.frame_id = "world_ned"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.id = idx * 3
            marker.header.stamp = rospy.Time.now()
            marker.pose.position.x = self.xk[self.arucos.index(arucos_num) + 3]
            marker.pose.position.y = self.xk[self.arucos.index(arucos_num) + 4]
            marker.pose.position.z = -0.1
            marker.pose.orientation.w = 0
            marker.scale.x = 0.08
            marker.scale.y = 0.08
            marker.scale.z = 0.08
            marker.color.r = 0.4
            marker.color.a = 0.3
            marker.color.g = 0.9
            ma.markers.append(marker)

            marker_text = Marker()
            marker_text.header.frame_id = "world_ned"
            marker_text.type = marker_text.TEXT_VIEW_FACING
            marker_text.action = marker_text.ADD
            marker_text.id = idx * 3 + 1
            marker_text.header.stamp = rospy.Time.now()
            marker_text.pose.position.x = self.xk[self.arucos.index(arucos_num) + 3]
            marker_text.pose.position.y = self.xk[self.arucos.index(arucos_num) + 4]
            marker_text.pose.position.z = -2.0 + 1.0
            marker_text.pose.orientation.w = -0.2
            marker_text.scale.z = 0.2
            marker_text.color.r = 0.4
            marker_text.color.a = 0.3
            marker_text.color.g = 0.9
            marker_text.text = "ArUco " + str(arucos_num)
            ma.markers.append(marker_text)

            # Add uncertainty ellipse
            uncertainty_marker = Marker()
            uncertainty_marker.header.frame_id = "world_ned"
            uncertainty_marker.type = uncertainty_marker.LINE_LIST
            uncertainty_marker.action = uncertainty_marker.ADD
            uncertainty_marker.id = idx * 3 + 2
            uncertainty_marker.header.stamp = rospy.Time.now()
            uncertainty_marker.pose.position.x = self.xk[self.arucos.index(arucos_num) + 3]
            uncertainty_marker.pose.position.y = self.xk[self.arucos.index(arucos_num) + 4]
            uncertainty_marker.pose.position.z = -0.1
            uncertainty_marker.pose.orientation.w = 0

            # Compute uncertainty ellipse parameters (assuming 2D Gaussian distribution)
            cov_matrix = np.array([[self.Pk[self.arucos.index(arucos_num) + 3, self.arucos.index(arucos_num) + 3], self.Pk[self.arucos.index(arucos_num) + 3, self.arucos.index(arucos_num) + 4]],
                                [self.Pk[self.arucos.index(arucos_num) + 4, self.arucos.index(arucos_num) + 3], self.Pk[self.arucos.index(arucos_num) + 4, self.arucos.index(arucos_num) + 4]]])

            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            major_axis_length = 2 * np.sqrt(4.605 * eigenvalues[0])  # 5.991 for 95% confidence ellipse
            minor_axis_length = 2 * np.sqrt(4.605 * eigenvalues[1])  # 4.605 for 68% confidence ellipse

            # Generate ellipse points
            theta = np.linspace(0, 2 * np.pi, 50)
            x = major_axis_length * np.cos(theta)
            y = minor_axis_length * np.sin(theta)
           # Apply rotation to ellipse points
            ellipse_points = np.dot(np.array([x, y]).T, eigenvectors.T) #+ np.array([[uncertainty_marker.pose.position.x, uncertainty_marker.pose.position.y]]).reshape(1, 2)

            # Set uncertainty marker properties
            uncertainty_marker.scale.x = 0.01  # Line width
            uncertainty_marker.color.r = 0.4
            uncertainty_marker.color.a = 0.3
            uncertainty_marker.color.g = 0.9

            # Set uncertainty marker points
            for i in range(len(theta) - 1):
                p1 = Point()
                p1.x = ellipse_points[i][0]
                p1.y = ellipse_points[i][1]
                p1.z = -0.1
                uncertainty_marker.points.append(p1)

                p2 = Point()
                p2.x = ellipse_points[i + 1][0]
                p2.y = ellipse_points[i + 1][1]
                p2.z = -0.1
                uncertainty_marker.points.append(p2)

            # Connect the last point with the first point to complete the ellipse
            p1 = Point()
            p1.x = ellipse_points[-1][0]
            p1.y = ellipse_points[-1][1]
            p1.z = -0.1
            uncertainty_marker.points.append(p1)

            p2 = Point()
            p2.x = ellipse_points[0][0]
            p2.y = ellipse_points[0][1]
            p2.z = -0.1
            uncertainty_marker.points.append(p2)

            # Add the uncertainty marker to the MarkerArray
            ma.markers.append(uncertainty_marker)

        self.arucos_belief_pub.publish(ma)

    def T_R_W(self, ran, azi):
        """
        Converts polar coordinates of a robot to Cartesian coordinates in the world frame.
        
        Args:
            ran (float): Range in polar coordinates (distance from the origin).
            azi (float): Azimuth angle in polar coordinates (angle with respect to the x-axis).
            
        Returns:
            tuple: Cartesian coordinates (x, y) in the world frame.
        """
            
        # Convert polar coordinates to Cartesian coordinates in the robot frame
        x_predict = ran * math.cos(azi)  # x-coordinate prediction
        y_predict = ran * math.sin(azi)  # y-coordinate prediction

        # Convert Cartesian coordinates from the robot frame to the world frame
        x_wo = x_predict * math.cos(self.xk[2]) - y_predict * math.sin(self.xk[2]) + self.xk[0]  # world x-coordinate
        y_wo = x_predict * math.sin(self.xk[2]) + y_predict * math.cos(self.xk[2]) + self.xk[1]  # world y-coordinate

        return x_wo, y_wo
        
    def markerpub_z(self,x,y,id):
        ma = MarkerArray()

        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.id = 1
        marker.header.stamp = rospy.Time.now()
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = -0.3
        marker.pose.orientation.w = 0
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        marker.color.r = 0.9
        marker.color.a = 0.3
        marker.color.g = 0.4
        ma.markers.append(marker)

        marker_text = Marker()
        marker_text.header.frame_id = "world_ned"
        marker_text.type = marker_text.TEXT_VIEW_FACING
        marker_text.action = marker_text.ADD
        marker_text.id = 2

        marker_text.header.stamp = rospy.Time.now()
        marker_text.pose.position.x = x
        marker_text.pose.position.y = y
        marker_text.pose.position.z = -2.0 + 1.0
        marker_text.pose.orientation.w = -0.4

        marker_text.scale.z = 0.2

        marker_text.color.r = 0.9
        marker_text.color.a = 0.3
        marker_text.color.g = 0.2


        marker_text.text = "Aruco from z "+str((id))

        ma.markers.append(marker_text)
                    
        self.arucos_z_pub.publish(ma)

    def markerpub_h(self,x,y,id):
        ma = MarkerArray()

        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.id = 1
        marker.header.stamp = rospy.Time.now()
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = -0.5
        marker.pose.orientation.w = 0
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        marker.color.r = 0.4
        marker.color.a = 0.3
        marker.color.b = 0.9
        ma.markers.append(marker)

        marker_text = Marker()
        marker_text.header.frame_id = "world_ned"
        marker_text.type = marker_text.TEXT_VIEW_FACING
        marker_text.action = marker_text.ADD
        marker_text.id = 2

        marker_text.header.stamp = rospy.Time.now()
        marker_text.pose.position.x = x
        marker_text.pose.position.y = y
        marker_text.pose.position.z = - 0.6
        marker_text.pose.orientation.w = -0.2

        marker_text.scale.z = 0.2

        marker_text.color.r = 0.4
        marker_text.color.a = 0.3
        marker_text.color.b = 0.9


        marker_text.text = "Aruco from h "+str((id))

        ma.markers.append(marker_text)
                    
        self.arucos_h_pub.publish(ma)

    # def r_trajec(self):
    #     path_m = Marker()
    #     path_m.header.stamp = rospy.Time.now()
    #     path_m.header.frame_id = 'world_ned'
    #     path_m.type = Marker.LINE_STRIP
    #     path_m.scale.x = 0.08
    #     path_m.scale.y = 0.08
    #     path_m.scale.z = 0.08
    #     path_m.color.r = 0.4
    #     path_m.color.a = 0.3
    #     path_m.color.b = 0.9


    def robot_path(self):
        return

        self.path_msg.header.frame_id = "world_ned"
        self.path_msg.header.stamp = rospy.Time.now()        

        # Create a PoseStamped message to store the robot's position
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "world_ned"

        # Update the robot's position and add it to the path
        pose_msg.pose.position.x = self.xk[0]
        pose_msg.pose.position.y = self.xk[1]
        # pose_msg.pose.position.z = -0.2
        pose_msg.pose.orientation.w = self.xk[2]
        self.path_msg.poses.append(pose_msg)

        # Publish the path
        self.path_pub.publish(self.path_msg)

    def gt_path(self):
        return
        
        self.pathgt_msg.header.frame_id = "world_ned"
        self.pathgt_msg.header.stamp = rospy.Time.now()        

        # Create a PoseStamped message to store the robot's position
        posegt_msg = PoseStamped()
        posegt_msg.header.frame_id = "world_ned"
        # rospy.Time.now()

        # Update the robot's position and add it to the path
        posegt_msg.pose.position.x = self.current_pose[0]
        posegt_msg.pose.position.y = self.current_pose[1]
        # pose_msg.pose.position.z = -0.2
        posegt_msg.pose.orientation.w = self.current_pose[2]
        self.pathgt_msg.poses.append(posegt_msg)

        # Publish the path
        self.pathgt_pub.publish(self.pathgt_msg)

if __name__ == '__main__':

    rospy.init_node('aruco_slam')

    robot = Robot()

    rospy.spin()

# ===============================================================================


##################################################################################################