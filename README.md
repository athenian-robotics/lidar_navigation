# TurtleBot3 Navigation with Lidar Data

## Setup

Clone the *lidar_navigation* repo into *~/catkin_ws/src* with:
```bash
# On Ubuntu PC
$ cd ~/catkin_ws/src
$ git clone https://github.com/athenian-robotics/lidar_navigation.git
```

Install the required python modules with:
```bash
# On Ubuntu PC
$ cd ~/catkin_ws/src/lidar_navigation
$ pip install -r requirements.txt
```

Compile the ROS support for Contour messages with:
```bash
# On Ubuntu PC
$ cd ~/catkin_ws
$ catkin_make
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

Add a maze model to the empty TurtleBot3 world with:
1) Click on the *Insert* tab
2) Click on one of the models under *~/catkin_ws/src/lidar_navigation/models*
3) Align the maze model with the TurtleBot3 model
4) Orient the screen with *Ctrl+Shift* mouse movements

## Launch Nodes

Launch the lidar navigation nodes with:
```bash
# On Ubuntu PC
# To publish the interpretation of the lidar data 
$ rosrun lidar_navigation geometry_node.py
# To drive the robot
$ rosrun lidar_navigation teleop_node.py
# To view the contour and centroid data
$ rosrun lidar_navigation contour_node.py --plot_all
# To view the walls data
$ rosrun lidar_navigation walls_node.py --plot_all
```

Stop a crashed robot with: 
```bash
# On Ubuntu PC
$ roslaunch lidar_navigation stop_node.py
```
**Warning**: Stop the *teleop_node.py* node before running the *stop_node.py* node. 

After stopping a crashed robot, reset the robot to the original starting position 
with *Ctrl+R*.

## Manual robot control
Launch keyboard teleop control with: 
```bash
# On Ubuntu PC
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```
**Warning**: Do not run *turtlebot3_teleop_key.launch* at the same time as  *teleop_node.py*. 


## Node CLI Options

### geometry_node.py 

| Option           | Description                                        | Default        |
|:-----------------|----------------------------------------------------|----------------|
| --slice_size     | Slice size degrees                                 | 5              |
| --slice_offset   | Slice offset                                       | 0              |
| --max_mult       | Maximum distance multiplier                        | 1.1            |
| --publish_rate   | Publish rate                                       | 30             |
| --scan_topic     | LaserScan values topic name                        | /scan          |
| --contour_topic  | Contour value topic name                           | /contour       |
| --centroid_topic | Centroid Point value topic name                    | /centroid      |
| --publish_pc     | Publish point cloud values                         | false          |
| --pc_topic       | PointCloud2 values topic name                      | /pc2           |
| --verbose        | Enable debugging info                              | false          |
| -h, --help       | Summary of options                                 |                |

### teleop_node.py 

| Option           | Description                                        | Default        |
|:-----------------|----------------------------------------------------|----------------|
| --max_linear     | Maximum linear speed                               | .35            |
| --max_angular    | Maximum angular speed                              | 2.75           |
| --stop_angle     | Full stop angle                                    | 70             |
| --publish_rate   | Publish rate                                       | 30             |
| --centroid_topic | Centroid Point value topic name                    | /centroid      |
| --verbose        | Enable debugging info                              | false          |
| -h, --help       | Summary of options                                 |                |

### contour_node.py 

| Option           | Description                                        | Default        |
|:-----------------|----------------------------------------------------|----------------|
| --plot_all       | Plot all items                                     | false          |
| --plot_points    | Plot points                                        | false          |
| --plot_contour   | Plot contour                                       | false          |
| --plot_centroid  | Plot centroid                                      | false          |
| --plot_slices    | Plot slices                                        | false          |
| --plot_mult      | Maximum plot multiplier                            | 1.05           |
| --contour_topic  | Contour value topic name                           | /contour       |
| --template_file  | Template file name                                 | /html/plot-image.html |
| --http           | HTTP hostname:port                                 | localhost:8080 |
| --http_delay     | HTTP delay secs                                    | 0              |
| --http_verbose   | Enable verbose HTTP log                            | false          |
| --verbose        | Enable debugging info                              | false          |
| -h, --help       | Summary of options                                 |                |


### walls_node.py 

| Option           | Description                                        | Default        |
|:-----------------|----------------------------------------------------|----------------|
| --plot_all       | Plot all items                                     | false          |
| --plot_points    | Plot points                                        | false          |
| --plot_centroid  | Plot centroid                                      | false          |
| --plot_mult      | Maximum plot multiplier                            | 1.05           |
| --contour_topic  | Contour value topic name                           | /contour       |
| --template_file  | Template file name                                 | /html/plot-image.html |
| --http           | HTTP hostname:port                                 | localhost:8080 |
| --http_delay     | HTTP delay secs                                    | 0              |
| --http_verbose   | Enable verbose HTTP log                            | false          |
| --verbose        | Enable debugging info                              | false          |
| -h, --help       | Summary of options                                 |                |


## RViz

Launch RViz viewer to view */scan* and */pc2* values with: 
```bash
# On Ubuntu PC
$ roslaunch turtlebot3_bringup turtlebot3_model.launch
```

## Creating a maze with gzmaze

Instructions on creating a maze with [gzmaze](https://github.com/athenian-robotics/gzmaze) are [here](./gzmaze.md)
