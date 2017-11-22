# TurtleBot3 Navigation with Lidar Data

## Setup

Install the required python modules with:
```bash
$ pip install -r requirements.txt
```

Compile the ROS support for InnerCountour messages with:
```bash
$ cd ~/catkin_ws
$ catlin_make
```

## Start a Turtlebot3

Start a physical Turtlebot3 with:
```bash
# On TurtleBot3
$ roslaunch turtlebot3_bringup turtlebot3_robot.launch 
```

Start a simulated TurtleBot3 with:
```bash
# On Ubuntu PC
$ roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch 
```

Insert a maze model into Gazebo with:
1) Click on the *Insert* tab.
2) Click on one of the models under ~/catkin_ws/src/liddar_navigation/models.
3) Align the maze model with the TurtleBot3 model.
4) Orient the screen with *ctrl-shift* mouse movements.

## Launch Nodes

Launch the lidar navigation nodes with:
```bash
$ rosrun lidar_navigation geometry_node.py
$ rosrun lidar_navigation image_node.py
$ rosrun lidar_navigation teleop_node.py
```

To stop a crashed robot, open keyboard teleop control node with: 
```bash
# On Ubuntu PC
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

## RViz

Launch RViz viewer to view */scan* and */pc2* values with: 
```bash
# On Ubuntu PC
$ roslaunch turtlebot3_bringup turtlebot3_model.launch
```

## Creating a maze with gzmaze

Instructions on creating a maze with [gzmaze](https://github.com/athenian-robotics/gzmaze) are [here](./gzmaze.md)
