#!/usr/bin/env python

import time
from threading import Lock

import rospy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist

import cli_args  as cli
from cli_args import setup_cli_args
from constants import LOG_LEVEL
from point2d import Point2D
from utils import new_twist
from utils import setup_logging


class LidarTeleop(object):
    def __init__(self, max_linear=.35, max_angular=2.75, vel_topic="/cmd_vel", centroid_topic="/centroid"):
        self.__max_linear = max_linear
        self.__max_angular = max_angular

        self.__curr_vals_lock = Lock()
        self.__curr_centroid = None
        self.__data_available = False

        rospy.loginfo("Publishing Twist msgs to topic {}".format(vel_topic))
        self.__vel_pub = rospy.Publisher(vel_topic, Twist, queue_size=5)

        rospy.loginfo("Subscribing to centroid topic {}".format(centroid_topic))
        self.__scan_sub = rospy.Subscriber(centroid_topic, Point, self.on_centroid)

    def on_centroid(self, centroid):
        with self.__curr_vals_lock:
            self.__curr_centroid = centroid
            self.__data_available = True

    def perform_teleop(self):
        try:
            while True:
                if not self.__data_available:
                    time.sleep(0.1)
                    continue

                with self.__curr_vals_lock:
                    # Convert the centroid from a ROS Point to a Point2D
                    centroid = Point2D(self.__curr_centroid.x, self.__curr_centroid.y)
                    self.__data_available = False

                # Calculate the linear and angular values proportional to the heading
                # In effect, slow down linear and speed up angular if a wide angle turn is necessary
                # and speed up linear and slow down angular if going straight
                full_stop_angle = 70
                if abs(centroid.heading) >= full_stop_angle:
                    linear = 0
                else:
                    linear = ((90.0 - abs(centroid.heading)) / 90.0) * self.__max_linear

                angular = -1 * ((centroid.heading / 90.0) * self.__max_angular)

                rospy.loginfo("Linear: {} Angular: {}".format(round(linear, 2), round(angular, 2)))

                # Publish Twist value
                twist = new_twist(linear, angular)
                self.__vel_pub.publish(twist)

        finally:
            # Stop robot when stopped
            rospy.loginfo("Sending stop value")
            for i in range(5):
                self.__vel_pub.publish(new_twist(0, 0))
            rospy.sleep(1)


if __name__ == '__main__':
    # Parse CLI args
    args = setup_cli_args(cli.log_level)

    # Setup logging
    setup_logging(level=args[LOG_LEVEL])

    rospy.init_node('teleop_node')

    rospy.loginfo("Running")

    try:
        LidarTeleop(max_linear=.35, max_angular=2.75).perform_teleop()
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        pass

    rospy.loginfo("Exiting")
