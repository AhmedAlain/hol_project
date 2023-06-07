# Hands-on-localisation Project


## Members of the goup:

This project has been carried out by:

* Ahmed Alghfeli
* Hassan Alhosani
* Reem Almeheri

## Aim of the package:

This package implements a feature-based SLAM system on the Kobuki robot and is simulated on Stonefish.


## Prerequisites

Before using this package, make sure you have the following dependencies installed:

* Python 3
* ROS
* Rviz and its plugin
* stonefish
* koubuki and its packages for stonefish
* opencv-contrib-python version 3.4.17.61
* numpy version 1.24.1


## Usage Instructions:
This package includes three launch files. The first one is for the Kobuki version on Stonefish, the second one is for the Turtlebot version, which includes the arm. The last launch file is for the real robot. In order for this to work properly, it needs to be in the same workspace as the Stonefish ROS, Turtlebot, Kobuki, and SwiftPro packages.


For Kobuki on Stonefish:

1) Launch the Kobuki-based SLAM system:

```bash
roslaunch hol_project koubuki_hol_slam.launch
```

2) (Optional) If you want to move the Kobuki:

```bash
rosrun hol_project controller_koubuki.py
```

For Turtlebot with an arm:

1) Launch the Turtlebot-based SLAM system:

```bash
roslaunch hol_project turtlebot_hol_ekf.py
```
2) (Optional) If you want to move the Turtlebot:

```bash
rosrun hol_project controller_turtlebot.py
```
For usage with a real robot:

```bash
roslaunch hol_project hol_ekf_robot.launch
```