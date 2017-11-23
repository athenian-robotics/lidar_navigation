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

Start a simulated TurtleBot3 in Gazebo with:
```bash
# On Ubuntu PC
$ roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch 
```

Insert a maze model into Gazebo with:
1) Click on the *Insert* tab.
2) Click on one of the models under ~/catkin_ws/src/lidar_navigation/models.
3) Align the maze model with the TurtleBot3 model.
4) Orient the screen with *ctrl-shift* mouse movements.

## Launch Nodes

Launch the lidar navigation nodes with:
```bash
$ rosrun lidar_navigation geometry_node.py
$ rosrun lidar_navigation teleop_node.py
$ rosrun lidar_navigation image_node.py
```

To stop a crashed robot, open keyboard teleop control node with: 
```bash
# On Ubuntu PC
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```
**Warning**: Do not run the keyboard teleop node at the same time as teleop_node.py. Two nodes
sending values to */cmd_vel* will cause problems. 


##  Node CLI Options

### geometry_node.py 

| Option           | Description                                        | Default        |
|:-----------------|----------------------------------------------------|----------------|
| --slice_size     | Slice size degrees                                 | 5              |
| --max_mult       | Maximum distance multiplier                        | 1.1            |
| --publish_rate   | Publish rate                                       | 30             |
| --scan_topic     | LaserScan values topic name                        | /scan          |
| --contour_topic  | InnerContour value topic name                      | /contour       |
| --centroid_topic | Centroid Point value topic name                    | /centroid      |
| --publish_pc     | Publish point cloud values                         | false          |
| --pc_topic       | PointCloud2 values topic name                      | /pc2           |
| --verbose        | Enable debugging info                              | false          |

### teleop_node.py 

| Option           | Description                                        | Default        |
|:-----------------|----------------------------------------------------|----------------|
| --max_linear     | Maximum linear speed                               | .35            |
| --max_angular    | Maximum angular speed                              | 2.75           |
| --stop_angle     | Full stop angle                                    | 70             |
| --publish_rate   | Publish rate                                       | 30             |
| --centroid_topic | Centroid Point value topic name                    | /centroid      |
| --verbose        | Enable debugging info                              | false          |

### image_node.py 

| Option           | Description                                        | Default        |
|:-----------------|----------------------------------------------------|----------------|
| --plot_all       | Plot all items                                     | false          |
| --plot_points    | Plot points                                        | false          |
| --plot_contour   | Plot contour                                       | false          |
| --plot_slices    | Plot slices                                        | false          |
| --plot_mult      | Maximum plot multiplier                            | 1.05           |
| --contour_topic  | InnerContour value topic name                      | /contour       |
| --verbose        | Enable debugging info                              | false          |

## RViz

Launch RViz viewer to view */scan* and */pc2* values with: 
```bash
# On Ubuntu PC
$ roslaunch turtlebot3_bringup turtlebot3_model.launch
```

## Creating a maze with gzmaze

Instructions on creating a maze with [gzmaze](https://github.com/athenian-robotics/gzmaze) are [here](./gzmaze.md)
