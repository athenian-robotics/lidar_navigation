#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

import cli_args  as cli
from cli_args import setup_cli_args
from constants import LOG_LEVEL
from utils import new_twist
from utils import setup_logging

if __name__ == '__main__':
    # Parse CLI args
    args = setup_cli_args(cli.log_level)

    # Setup logging
    setup_logging(level=args[LOG_LEVEL])

    rospy.init_node('stop_node')

    rospy.loginfo("Running")

    vel_topic = "/cmd_vel"
    rospy.loginfo("Publishing Twist vals to topic: {}".format(vel_topic))
    vel_pub = rospy.Publisher(vel_topic, Twist, queue_size=5)
    for i in range(10):
        vel_pub.publish(new_twist(0, 0))
    rospy.sleep(1)

    rospy.loginfo("Exiting")
