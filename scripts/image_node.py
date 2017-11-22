#!/usr/bin/env python

import cStringIO

import matplotlib

# Execute matplotlib.use('Agg') if using image_server
# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from threading import Lock

import rospy
import time
import cli_args  as cli
from constants import LOG_LEVEL
from cli_args import setup_cli_args
from constants import HTTP_DELAY_SECS, HTTP_HOST, HTTP_FILE, HTTP_VERBOSE
from constants import PLOT_ALL, PLOT_CONTOUR, PLOT_POINTS, PLOT_SLICES, PLOT_MULT
from image_server import ImageServer
from utils import setup_logging
from lidar_navigation.msg import InnerContour
from point2d import Point2D
from slice import Slice


class LidarImage(object):
    def __init__(self,
                 image_server=None,
                 plot_all=False,
                 plot_contour=False,
                 plot_points=False,
                 plot_slices=False,
                 plot_mult=1.05,
                 contour_topic="/contour"):
        self.__plot_all = plot_all
        self.__plot_points = plot_points
        self.__plot_contour = plot_contour
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

        rospy.loginfo("Subscribing to InnerContour topic: {}".format(contour_topic))
        self.__contour_sub = rospy.Subscriber(contour_topic, InnerContour, self.on_contour)

    def on_contour(self, contour):
        # Pass the values to be plotted
        with self.__curr_vals_lock:
            self.__max_dist = contour.max_dist
            self.__slice_size = contour.slice_size
            self.__centroid = Point2D(contour.centroid.x, contour.centroid.y)
            self.__all_points = [Point2D(p.x, p.y) for p in contour.all_points]
            self.__nearest_points = [Point2D(p.x, p.y) for p in contour.nearest_points]
            self.__data_available = True

    def generate_image(self):
        while True:
            if not self.__data_available:
                time.sleep(0.1)
                continue

            with self.__curr_vals_lock:
                max_dist = self.__max_dist
                slice_size = self.__slice_size
                centroid = self.__centroid
                all_points = self.__all_points
                nearest_points = self.__nearest_points
                self.__data_available = False

            # Initialize plot
            plt.figure(figsize=(8, 8), dpi=80)
            plt.grid(True)

            # Plot robot center
            plt.plot([0], [0], 'r^', markersize=8.0)

            # Plot centroid
            plt.plot([centroid.x], [centroid.y], 'g^', markersize=8.0)

            # Plot point cloud
            if self.__plot_points or self.__plot_all:
                plt.plot([p.x for p in all_points], [p.y for p in all_points], 'ro', markersize=2.0)

            # Plot inner contour
            if self.__plot_contour or self.__plot_all:
                icx = [p.x for p in nearest_points]
                icy = [p.y for p in nearest_points]
                plt.plot(icx, icy, 'b-')
                plt.plot(icx, icy, 'go', markersize=4.0)

            # Plot slices
            if self.__plot_slices or self.__plot_all:
                slices = [Slice(v, v + slice_size) for v in range(0, 180, slice_size)]
                for s in slices:
                    plt.plot([s.begin_point(max_dist).x, 0], [s.begin_point(max_dist).y, 0], 'b-')
                plt.plot([slices[-1].end_point(max_dist).x, 0], [slices[-1].end_point(max_dist).y, 0], 'b-')

            # Write Heading
            c = Point2D(centroid.x, centroid.y)
            plt.title("Heading: {} Distance: {}".format(c.heading, round(c.dist, 2)))

            # Plot axis
            plt.axis(
                [(-1 * max_dist) * self.__plot_mult, max_dist * self.__plot_mult, - 0.05, max_dist * self.__plot_mult])

            if self.__image_server is not None:
                sio = cStringIO.StringIO()
                plt.savefig(sio, format="jpg")
                self.__image_server.image = sio.getvalue()
                sio.close()
            else:
                plt.show()

            # Close resources
            plt.close()


if __name__ == '__main__':
    # Parse CLI args
    args = setup_cli_args(cli.contour_topic,
                          cli.plot_all,
                          cli.plot_points,
                          cli.plot_contour,
                          cli.plot_slices,
                          cli.plot_mult,
                          cli.log_level)

    # Setup logging
    setup_logging(level=args[LOG_LEVEL])

    rospy.init_node('contour_node')

    image_server = ImageServer(template_file=args[HTTP_FILE],
                               http_host=args[HTTP_HOST],
                               http_delay_secs=args[HTTP_DELAY_SECS],
                               http_verbose=args[HTTP_VERBOSE])

    rospy.loginfo("Running")

    try:
        image_server.start()
        image = LidarImage(image_server=image_server,
                           plot_all=args[PLOT_ALL],
                           plot_points=args[PLOT_POINTS],
                           plot_contour=args[PLOT_CONTOUR],
                           plot_slices=args[PLOT_SLICES],
                           plot_mult=args[PLOT_MULT])
        image.generate_image()
    except KeyboardInterrupt:
        pass
    finally:
        image_server.stop()

    rospy.loginfo("Exiting")
