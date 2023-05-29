#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
import tf
# from tf import TransformListener


class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_2", Image)
        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber(
        #     "/kobuki/sensors/realsense/color/image_color", Image, self.callback)

        self.image_sub = rospy.Subscriber("turtlebot/kobuki/sensors/realsense/color/image_color", Image, self.callback)
        self.camera_info_sub = rospy.Subscriber("turtlebot/kobuki/sensors/realsense/color/camera_info", CameraInfo, self.camera_info_callback)

        self.measured_pub = rospy.Publisher("measurd_data", Float32MultiArray)
        self.aruco_dict = cv2.aruco.Dictionary_get(
            cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        # self.camera_matrix = np.array([[1396.8086675255468, 0.0, 960.0], [
        #                               0.0, 1396.8086675255468, 540.0], [0.0, 0.0, 1.0]])
        # self.dist_coeffs = np.zeros((5, 1))
        self.camera_matrix = None
        self.dist_coeffs = None
        self.ids = None
        self.marker_size_mymap = 0.06 # size of ArUco marker in meters
        self.marker_size_profmap = 0.15 # size of ArUco marker in meters

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Detect the ArUco markers in the image
        corners, self.ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dict, parameters=self.aruco_params)

        if self.ids is not None:
            # Draw the detected markers and IDs on the image
            cv2.aruco.drawDetectedMarkers(cv_image, corners, self.ids)

            # Get the position and orientation of each detected marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size_profmap, self.camera_matrix, self.dist_coeffs)

            # Draw the axes of each detected marker
            for i in range(len(self.ids)):
                cv2.aruco.drawAxis(cv_image, self.camera_matrix,
                                   self.dist_coeffs, rvecs[i], tvecs[i], self.marker_size_profmap)

                # Get the position and orientation of the marker
                rvec = rvecs[i]
                tvec = tvecs[i]

                # Convert the rotation vector to a rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                roll_c = np.pi/2.0
                pitch_c = 0
                yaw_c = np.pi/2.0

                # Define the rotation vector from the roll, pitch, and yaw angles
                rvec_c = np.array([roll_c, pitch_c, yaw_c])
                R_c, _ = cv2.Rodrigues(rvec_c)
                # print("R_c",R_c)

                T_r_c_ = np.array([[0, 0, 1,  0.122],
                                   [1, 0, 0, -0.033],
                                   [0, 1, 0,  0.082],
                                   [0, 0, 0,      1]])
                # T_r_c_[:3, :3] = R_c
                T_c_m = np.eye(4)
                T_c_m[:3, :3] = R
                
                T_c_m[:3, 3] = tvec
                T_r_m = T_r_c_@ T_c_m
                x, y, z, _ = T_r_m[:, 3]
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                # elev = np.arctan2(z,np.sqrt(x**2 + y**2))
                elev = np.arcsin(z / r)
                measured = [self.ids[i], r, phi, elev]
                self.measured_pub.publish(Float32MultiArray(data=measured))
                # print(
                #     f"--Marker ID {self.ids[i]} - Position: {x,y,z} - Orientation: {rvec} - Range: {r} - Azimuth: {phi} - Elevation:{elev}")

                # aruco_id = IDs[i]
        # cv2.namedWindow('image',WINDOW_NORMAL)
        scale_percent = 30 # percent of original size
        width = int(cv_image.shape[1] * scale_percent / 100)
        height = int(cv_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(cv_image, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("Image window", resized)
        cv2.waitKey(3)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
