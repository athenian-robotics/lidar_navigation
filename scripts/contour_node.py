#!/usr/bin/env python

import cStringIO

import matplotlib

# Execute matplotlib.use('Agg') if using image_server
# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from threading import Lock
from threading import Thread

import rospy
import time
import cli_args  as cli
from constants import LOG_LEVEL
from cli_args import setup_cli_args
from constants import HTTP_DELAY_SECS, HTTP_HOST, TEMPLATE_FILE, HTTP_VERBOSE
from constants import PLOT_ALL, PLOT_CONTOUR, PLOT_CENTROID, PLOT_POINTS, PLOT_SLICES, MAX_AXIS_MULT
from image_server import ImageServer
from utils import setup_logging
from lidar_navigation.msg import Contour
from point2d import Point2D
from point2d import Origin
from slice import Slice


class LidarImage(object):
    def __init__(self,
                 image_server=None,
                 plot_all=False,
                 plot_contour=False,
                 plot_centroid=False,
                 plot_points=False,
                 plot_slices=False,
                 max_axis_mult=1.05,
                 contour_topic="/contour"):
        self.__plot_all = plot_all
        self.__plot_points = plot_points
        self.__plot_contour = plot_contour
        self.__plot_centroid = plot_centroid
        self.__plot_slices = plot_slices
        self.__max_axis_mult = max_axis_mult
        self.__image_server = image_server

        self.__curr_vals_lock = Lock()
        self.__curr_msg = None
        self.__data_available = False
        self.__stopped = False

        rospy.loginfo("Subscribing to Contour topic {}".format(contour_topic))
        self.__contour_sub = rospy.Subscriber(contour_topic, Contour, self.on_msg)

    def on_msg(self, contour_msg):
        with self.__curr_vals_lock:
            self.__curr_msg = contour_msg
            self.__data_available = True

    def generate_image(self):
        while not self.__stopped:
            if not self.__data_available:
                time.sleep(0.1)
                continue

            with self.__curr_vals_lock:
                contour_msg = self.__curr_msg
                self.__data_available = False

            max_dist = contour_msg.max_dist
            slice_size = contour_msg.slice_size
            centroid = Point2D(contour_msg.centroid.x, contour_msg.centroid.y)
            all_points = [Point2D(p.x, p.y) for p in contour_msg.all_points]
            nearest_points = [Point2D(p.x, p.y) for p in contour_msg.nearest_points]

            # Initialize plot
            plt.figure(figsize=(8, 8), dpi=80)
            plt.grid(True)

            # Plot robot center
            plt.plot([0], [0], 'r^', markersize=8.0)

            # Plot centroid and write heading
            if self.__plot_centroid or self.__plot_all:
                c = Point2D(centroid.x, centroid.y)
                plt.title("Heading: {} Distance: {}".format(c.heading, round(c.origin_dist, 2)))
                plt.plot([centroid.x], [centroid.y], 'g^', markersize=8.0)

            # Plot point cloud
            if self.__plot_points or self.__plot_all:
                plt.plot([p.x for p in all_points], [p.y for p in all_points], 'ro', markersize=2.0)

            # Plot contour
            if self.__plot_contour or self.__plot_all:
                nearest_with_origin = [Origin] + nearest_points + [Origin]
                icx = [p.x for p in nearest_with_origin]
                icy = [p.y for p in nearest_with_origin]
                plt.plot(icx, icy, 'b-')
                plt.plot(icx, icy, 'go', markersize=4.0)

            # Plot slices
            if self.__plot_slices or self.__plot_all:
                slices = [Slice(v, v + slice_size) for v in range(0, 180, slice_size)]
                linestyle = 'r:'
                for s in slices:
                    plt.plot([s.begin_point(max_dist).x, 0], [s.begin_point(max_dist).y, 0], linestyle)
                plt.plot([slices[-1].end_point(max_dist).x, 0], [slices[-1].end_point(max_dist).y, 0], linestyle)

            # Plot axis
            dist = max_dist * self.__max_axis_mult
            plt.axis([-1 * dist, dist, - 0.05, dist])

            if self.__image_server is not None:
                sio = cStringIO.StringIO()
                plt.savefig(sio, format="jpg")
                self.__image_server.image = sio.getvalue()
                sio.close()
            else:
                plt.show()

            # Close resources
            plt.close()

    def stop(self):
        self.__stopped = True


if __name__ == '__main__':
    # Parse CLI args
    args = setup_cli_args(cli.plot_all,
                          cli.plot_points,
                          cli.plot_contour,
                          cli.plot_centroid,
                          cli.plot_slices,
                          cli.max_axis_mult,
                          cli.contour_topic,
                          ImageServer.args,
                          cli.log_level)

    # Setup logging
    setup_logging(level=args[LOG_LEVEL])

    rospy.init_node('contour_node')

    image_server = ImageServer(template_file=args[TEMPLATE_FILE],
                               http_host=args[HTTP_HOST],
                               http_delay_secs=args[HTTP_DELAY_SECS],
                               http_verbose=args[HTTP_VERBOSE])

    image_server.start()
    image = LidarImage(image_server=image_server,
                       plot_all=args[PLOT_ALL],
                       plot_points=args[PLOT_POINTS],
                       plot_contour=args[PLOT_CONTOUR],
                       plot_centroid=args[PLOT_CENTROID],
                       plot_slices=args[PLOT_SLICES],
                       max_axis_mult=args[MAX_AXIS_MULT])

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
