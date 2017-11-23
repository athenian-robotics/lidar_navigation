# Generating Mazes

The [gzmaze](https://github.com/athenian-robotics/gzmaze) repo provides an easy way to generate
a Gazebo maze that is compatible with the TurtleBot3 model. 


## Setup

```bash
$ cd ~/catkin_ws/src
$ git clone https://github.com/athenian-robotics/gzmaze.git
$ cd gzmaze
$ mkdir build
$ cd build
$ cmake ..
$ make
```

Add the following to *.gazebo/gui.ini*:
```bash
[overlay_plugins]
filenames=libregenerate_widget.so
```

## Running Gazebo to Generate a Model

```bash
$ cd ~/catkin_ws/src/gzmaze/
$ ./setup.sh
```
 
Once Gazebo starts up, click *Browse For Maze File* to choose your maze definition 
and then click *Generate Model* to generate the maze model. 

To save a maze as a model, right click on the maze and select *Edit model*.
Warning: the Gazebo process will be unresponsive for a couple minutes until it
enters the Model Editor. 

Once you enter the Model Editor, save the model by clicking on the menu item *File->Save As*.

Exit the Model Editor by clicking on *File->Exit Model Editor*.
