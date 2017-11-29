#!/usr/bin/env python

import time
from threading import Lock
from threading import Thread

import rospy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist

import cli_args  as cli
from cli_args import setup_cli_args
from constants import CENTROID_TOPIC_DEFAULT, VEL_TOPIC_DEFAULT
from constants import LOG_LEVEL
from constants import MAX_LINEAR, MAX_ANGULAR, CENTROID_TOPIC, STOP_ANGLE_DEFAULT, PUBLISH_RATE_DEFAULT
from constants import VEL_TOPIC, STOP_ANGLE, PUBLISH_RATE, MAX_LINEAR_DEFAULT, MAX_ANGULAR_DEFAULT
from point2d import Point2D
from utils import new_twist
from utils import setup_logging


class LidarTeleop(object):
    def __init__(self,
                 max_linear=MAX_LINEAR_DEFAULT,
                 max_angular=MAX_ANGULAR_DEFAULT,
                 full_stop_angle=STOP_ANGLE_DEFAULT,
                 publish_rate=PUBLISH_RATE_DEFAULT,
                 vel_topic=VEL_TOPIC_DEFAULT,
                 centroid_topic=CENTROID_TOPIC_DEFAULT):
        self.__max_linear = max_linear
        self.__max_angular = max_angular
        self.__full_stop_angle = full_stop_angle

        self.__rate = rospy.Rate(publish_rate)
        self.__curr_vals_lock = Lock()
        self.__curr_msg = None
        self.__data_available = False
        self.__stopped = False

        rospy.loginfo("Publishing Twist vals to topic {}".format(vel_topic))
        self.__vel_pub = rospy.Publisher(vel_topic, Twist, queue_size=5)

        rospy.loginfo("Subscribing to Point topic {}".format(centroid_topic))
        self.__scan_sub = rospy.Subscriber(centroid_topic, Point, self.on_msg)

    def on_msg(self, centroid_msg):
        with self.__curr_vals_lock:
            self.__curr_msg = centroid_msg
            self.__data_available = True

    def perform_teleop(self):
        try:
            while not self.__stopped:
                if not self.__data_available:
                    time.sleep(0.1)
                    continue

                with self.__curr_vals_lock:
                    centroid_msg = self.__curr_msg
                    self.__data_available = False

                # Convert the centroid from a ROS Point to a Point2D
                centroid = Point2D(centroid_msg.x, centroid_msg.y)

                # Calculate the linear and angular values proportional to the heading
                # In effect, slow down linear and speed up angular if a wide angle turn is necessary
                # and speed up linear and slow down angular if going straight
                if abs(centroid.heading) >= self.__full_stop_angle:
                    linear = 0
                else:
                    linear = ((90.0 - abs(centroid.heading)) / 90.0) * self.__max_linear

                angular = -1 * ((centroid.heading / 90.0) * self.__max_angular)

                rospy.loginfo("Linear: {} Angular: {}".format(round(linear, 2), round(angular, 2)))

                # Publish Twist value
                twist = new_twist(linear, angular)
                self.__vel_pub.publish(twist)

                self.__rate.sleep()

        except KeyboardInterrupt:
            # This will prevent callstack dump on exit with Ctrl+C
            pass

    def stop(self):
        self.__stopped = True

    def send_stop(self):
        # Stop robot when stopped
        rospy.loginfo("Sending stop value")
        self.__vel_pub.publish(new_twist(0, 0))


if __name__ == '__main__':
    # Parse CLI args
    args = setup_cli_args(cli.max_linear,
                          cli.max_angular,
                          cli.stop_angle,
                          cli.publish_rate,
                          cli.centroid_topic,
                          cli.vel_topic,
                          cli.log_level)

    # Setup logging
    setup_logging(level=args[LOG_LEVEL])

    rospy.init_node('teleop_node')

    teleop = LidarTeleop(max_linear=args[MAX_LINEAR],
                         max_angular=args[MAX_ANGULAR],
                         full_stop_angle=args[STOP_ANGLE],
                         publish_rate=args[PUBLISH_RATE],
                         centroid_topic=args[CENTROID_TOPIC],
                         vel_topic=args[VEL_TOPIC])

    rospy.loginfo("Running")

    try:
        # Running this in a thread will enable Ctrl+C exits
        Thread(target=teleop.perform_teleop).start()
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        teleop.stop()
        teleop.send_stop()

    rospy.loginfo("Exiting")
