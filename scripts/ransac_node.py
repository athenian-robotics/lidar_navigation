#!/usr/bin/env python

import cStringIO

import matplotlib

# Execute matplotlib.use('Agg') if using image_server
# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from threading import Lock
from threading import Thread

import random
import rospy
import sys
import time
import cli_args  as cli
from constants import LOG_LEVEL
from cli_args import setup_cli_args
from constants import HTTP_DELAY_SECS, HTTP_HOST, TEMPLATE_FILE, HTTP_VERBOSE
from constants import PLOT_ALL, PLOT_CENTROID, PLOT_POINTS, PLOT_SLICES, PLOT_MULT
from image_server import ImageServer
from utils import setup_logging
from lidar_navigation.msg import Contour
from point2d import Point2D
from slice import Slice


class LidarRansac(object):
    def __init__(self,
                 image_server=None,
                 iterations=40,
                 threshold=0.1 / 4,
                 plot_all=False,
                 plot_centroid=False,
                 plot_points=False,
                 plot_slices=False,
                 plot_mult=1.05,
                 contour_topic="/contour"):
        self.__iterations = iterations
        self.__threshold = threshold
        self.__plot_all = plot_all
        self.__plot_points = plot_points
        self.__plot_centroid = plot_centroid
        self.__plot_slices = plot_slices
        self.__plot_mult = plot_mult
        self.__image_server = image_server

        self.__curr_vals_lock = Lock()
        self.__all_points = []
        self.__nearest_points = []
        self.__max_dist = None
        self.__slice_size = None
        self.__centroid = None
        self.__data_available = False
        self.__stopped = False

        rospy.loginfo("Subscribing to Contour topic {}".format(contour_topic))
        self.__contour_sub = rospy.Subscriber(contour_topic, Contour, self.on_msg)

        random.seed(0)

    def on_msg(self, contour_msg):
        # Pass the values to be plotted
        with self.__curr_vals_lock:
            self.__max_dist = contour_msg.max_dist
            self.__slice_size = contour_msg.slice_size
            self.__centroid = Point2D(contour_msg.centroid.x, contour_msg.centroid.y)
            self.__all_points = [Point2D(p.x, p.y) for p in contour_msg.all_points]
            self.__nearest_points = [Point2D(p.x, p.y) for p in contour_msg.nearest_points]
            self.__data_available = True

    def generate_image(self):
        while not self.__stopped:
            if not self.__data_available:
                time.sleep(0.1)
                continue

            with self.__curr_vals_lock:
                max_dist = self.__max_dist
                slice_size = self.__slice_size
                centroid = self.__centroid
                all_points = self.__all_points
                self.__data_available = False

            if len(all_points) < 2:
                rospy.loginfo("Invalid all_points size: {}".format(len(all_points)))
                continue

            inliers = []
            outliers = []
            final_m = None
            final_b = None
            final_p0 = None
            final_p1 = None
            for i in range(self.__iterations):
                p0, p1 = random_pair(all_points)
                m, b = slopeYInt(p0, p1)

                iter_inliners = []
                iter_outliers = []
                for p in all_points:
                    dist = p.distance_to_line(p0, p1)
                    if dist <= self.__threshold:
                        iter_inliners.append(p)
                    else:
                        iter_outliers.append(p)

                if len(iter_inliners) > len(inliers):
                    inliers = iter_inliners
                    outliers = iter_outliers
                    final_m = m
                    final_b = b
                    final_p0 = p0
                    final_p1 = p1

            rospy.loginfo("Found wall with {} items slope: {} yint: {}".format(len(inliers), final_m, final_b))

            # Initialize plot
            plt.figure(figsize=(8, 8), dpi=80)
            plt.grid(True)

            # Plot robot center
            plt.plot([0], [0], 'r^', markersize=8.0)

            # Plot centroid and write heading
            if self.__plot_centroid or self.__plot_all:
                c = Point2D(centroid.x, centroid.y)
                plt.title("Heading: {} Distance: {}".format(c.heading, round(c.dist, 2)))
                plt.plot([centroid.x], [centroid.y], 'g^', markersize=8.0)

            # Plot point cloud
            if self.__plot_points or self.__plot_all:
                plt.plot([final_p0.x, final_p1.x], [final_p0.y, final_p1.y], 'r^', markersize=6.0)
                plt.plot([p.x for p in inliers], [p.y for p in inliers], 'go', markersize=2.0)
                # plt.plot([p.x for p in outliers], [p.y for p in outliers], 'ro', markersize=2.0)

            # Plot slices
            if self.__plot_slices or self.__plot_all:
                slices = [Slice(v, v + slice_size) for v in range(0, 180, slice_size)]
                linestyle = 'r:'
                for s in slices:
                    plt.plot([s.begin_point(max_dist).x, 0], [s.begin_point(max_dist).y, 0], linestyle)
                plt.plot([slices[-1].end_point(max_dist).x, 0], [slices[-1].end_point(max_dist).y, 0], linestyle)

            # Plot axis
            plt.axis(
                [(-1 * max_dist) * self.__plot_mult, max_dist * self.__plot_mult, - 0.05,
                 max_dist * self.__plot_mult])

            if self.__image_server is not None:
                sio = cStringIO.StringIO()
                plt.savefig(sio, format="jpg")
                self.__image_server.image = sio.getvalue()
                sio.close()
            else:
                plt.show()

            # Close resources
            plt.close()

            rospy.sleep(2)

    def stop(self):
        self.__stopped = True


def random_pair(points):
    cnt = len(points)
    if cnt < 2:
        return None, None
    while True:
        index0 = int(random.uniform(0, cnt))
        index1 = int(random.uniform(0, cnt))
        if index0 != index1:
            return points[index0], points[index1]


def slopeYInt(p0, p1):
    xdiff = p1.x - p0.x
    # Avoid div by zero problems by adding a little noise
    if xdiff == 0:
        xdiff = sys.float_info.epsilon
    m = (p1.y - p0.y) / xdiff
    y = p0.y - (p0.x * m)
    return m, y


if __name__ == '__main__':
    # Parse CLI args
    args = setup_cli_args(cli.plot_all,
                          cli.plot_points,
                          cli.plot_centroid,
                          cli.plot_slices,
                          cli.plot_mult,
                          cli.contour_topic,
                          ImageServer.args,
                          cli.log_level)

    # Setup logging
    setup_logging(level=args[LOG_LEVEL])

    rospy.init_node('ransac_node')

    image_server = ImageServer(template_file=args[TEMPLATE_FILE],
                               http_host=args[HTTP_HOST],
                               http_delay_secs=args[HTTP_DELAY_SECS],
                               http_verbose=args[HTTP_VERBOSE])

    image_server.start()
    image = LidarRansac(image_server=image_server,
                        plot_all=args[PLOT_ALL],
                        plot_points=args[PLOT_POINTS],
                        plot_centroid=args[PLOT_CENTROID],
                        plot_slices=args[PLOT_SLICES],
                        plot_mult=args[PLOT_MULT])

    rospy.loginfo("Running")

    try:
        # Running this in a thread will enable Ctrl+C exits
        Thread(target=image.generate_image).start()
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        image.stop()
        image_server.stop()

    rospy.loginfo("Exiting")
