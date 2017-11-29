#!/usr/bin/env python

import time
from threading import Lock
from threading import Thread

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point
from laser_geometry import LaserProjection
from lidar_navigation.msg import Contour
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from shapely.geometry import Polygon

import cli_args  as cli
from cli_args import setup_cli_args
from constants import LOG_LEVEL, SLICE_OFFSET_DEFAULT, MAX_DIST_MULT_DEFAULT, PUBLISH_PC_DEFAULT
from constants import SCAN_TOPIC, CONTOUR_TOPIC, CENTROID_TOPIC, PC_TOPIC, SLICE_SIZE_DEFAULT
from constants import SCAN_TOPIC_DEFAULT, CONTOUR_TOPIC_DEFAULT, CENTROID_TOPIC_DEFAULT, PC_TOPIC_DEFAULT
from constants import SLICE_SIZE, SLICE_OFFSET, MAX_DIST_MULT, PUBLISH_PC, PUBLISH_RATE, PUBLISH_RATE_DEFAULT
from point2d import Origin
from point2d import Point2D
from slice import Slice
from utils import setup_logging


class LidarGeometry(object):
    def __init__(self,
                 slice_size=SLICE_SIZE_DEFAULT,
                 slice_offset=SLICE_OFFSET_DEFAULT,
                 max_dist_mult=MAX_DIST_MULT_DEFAULT,
                 publish_rate=PUBLISH_RATE_DEFAULT,
                 publish_pc=PUBLISH_PC_DEFAULT,
                 scan_topic=SCAN_TOPIC_DEFAULT,
                 contour_topic=CONTOUR_TOPIC_DEFAULT,
                 centroid_topic=CENTROID_TOPIC_DEFAULT,
                 pc_topic=PC_TOPIC_DEFAULT):
        self.__slice_size = slice_size
        self.__slice_offset = slice_offset
        self.__max_dist_mult = max_dist_mult
        self.__publish_point_cloud = publish_pc

        self.__rate = rospy.Rate(publish_rate)
        self.__laser_projector = LaserProjection()
        self.__vals_lock = Lock()
        self.__curr_msg = None
        self.__data_available = False
        self.__stopped = False

        # Create Slices once and reset them on each iteration
        self.__slices = [Slice(v, v + self.__slice_size) for v in range(0, 180, self.__slice_size)]

        rospy.loginfo("Publishing Contour vals to topic {}".format(contour_topic))
        self.__contour_pub = rospy.Publisher(contour_topic, Contour, queue_size=5)

        rospy.loginfo("Publishing Point vals to topic {}".format(centroid_topic))
        self.__centroid_pub = rospy.Publisher(centroid_topic, Point, queue_size=5)

        if self.__publish_point_cloud:
            rospy.loginfo("Publishing PointCloud2 vals to topic {}".format(pc_topic))
            self.__pc_pub = rospy.Publisher(pc_topic, PointCloud2, queue_size=5)

        rospy.loginfo("Subscribing to LaserScan topic {}".format(scan_topic))
        self.__scan_sub = rospy.Subscriber(scan_topic, LaserScan, self.on_msg)

    def on_msg(self, scan_msg):
        # Pass the scan_msg
        with self.__vals_lock:
            self.__curr_msg = scan_msg
            self.__data_available = True

    def eval_points(self):
        try:
            while not self.__stopped:
                if not self.__data_available:
                    time.sleep(0.1)
                    continue

                with self.__vals_lock:
                    scan_msg = self.__curr_msg
                    self.__data_available = False

                # https://answers.ros.org/question/202787/using-pointcloud2-data-getting-xy-points-in-python/
                point_cloud = self.__laser_projector.projectLaser(scan_msg)

                # Publish point cloud data
                if self.__publish_point_cloud:
                    self.__pc_pub.publish(point_cloud)

                # Shift all points counter clockwise 90 degrees - switch x,y and multiply x by -1
                all_points = []
                for p in pc2.read_points(point_cloud, field_names=("x", "y", "z"), skip_nans=True):
                    x = -1 * p[1]
                    y = p[0]
                    # Track only points in front of robot -- NW and NE quadrants
                    if y >= 0:
                        all_points.append(Point2D(x, y))

                if len(all_points) == 0:
                    continue

                # Determine outer range of points
                max_dist = round(max([p.origin_dist for p in all_points]) * self.__max_dist_mult, 2)
                rospy.loginfo("Points: {} Max dist: {}".format(len(all_points), round(max_dist, 2)))

                # Reset all slices
                [s.reset(max_dist) for s in self.__slices]

                # Assign each slice
                for p in all_points:
                    # Do int division to determine which slice the point belongs to
                    slice_index = int(p.angle / self.__slice_size)
                    self.__slices[slice_index].add_point(p)

                # Adjust for slice_offset, which is a subset of slices
                nearest_points = [s.nearest for s in self.__slices]
                if self.__slice_offset > 0:
                    nearest_points = nearest_points[self.__slice_offset:-1 * self.__slice_offset]

                if len(nearest_points) == 0:
                    continue

                # Perform these outside of lock to prevent blocking on scan readings
                # Calculate contour and centroid
                nearest_with_origin = [Origin] + nearest_points + [Origin]
                icx = [p.x for p in nearest_with_origin]
                icy = [p.y for p in nearest_with_origin]
                polygon = Polygon(zip(icx, icy))
                poly_centroid = polygon.centroid

                # Convert to msgs
                c = Contour()
                c.max_dist = max_dist
                c.slice_size = self.__slice_size
                c.all_points = np.asarray([p.to_ros_point() for p in all_points])
                c.nearest_points = np.asarray([p.to_ros_point() for p in nearest_points])
                c.centroid = Point2D(poly_centroid.x, poly_centroid.y).to_ros_point()

                # Publish centroid and contour
                self.__centroid_pub.publish(c.centroid)
                self.__contour_pub.publish(c)

                self.__rate.sleep()

        except KeyboardInterrupt:
            # This will prevent callstack dump on exit with Ctrl+C
            pass

    def stop(self):
        self.__stopped = True


if __name__ == '__main__':
    # Parse CLI args
    args = setup_cli_args(cli.slice_size,
                          cli.slice_offset,
                          cli.max_dist_mult,
                          cli.publish_rate,
                          cli.scan_topic,
                          cli.contour_topic,
                          cli.centroid_topic,
                          cli.publish_pc,
                          cli.pc_topic,
                          cli.log_level)

    # Setup logging
    setup_logging(level=args[LOG_LEVEL])

    rospy.init_node('geometry_node')

    geometry = LidarGeometry(slice_size=args[SLICE_SIZE],
                             slice_offset=args[SLICE_OFFSET],
                             max_dist_mult=args[MAX_DIST_MULT],
                             publish_rate=args[PUBLISH_RATE],
                             publish_pc=args[PUBLISH_PC],
                             scan_topic=args[SCAN_TOPIC],
                             contour_topic=args[CONTOUR_TOPIC],
                             centroid_topic=args[CENTROID_TOPIC],
                             pc_topic=args[PC_TOPIC])

    rospy.loginfo("Running")

    try:
        # Running this in a thread will enable Ctrl+C exits
        Thread(target=geometry.eval_points).start()
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        geometry.stop()

    rospy.loginfo("Exiting")
