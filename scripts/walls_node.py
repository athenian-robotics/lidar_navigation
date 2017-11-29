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
import time
import cli_args  as cli
from constants import LOG_LEVEL
from cli_args import setup_cli_args
from constants import HTTP_DELAY_SECS, HTTP_HOST, TEMPLATE_FILE, HTTP_VERBOSE
from constants import PLOT_ALL, PLOT_CENTROID, PLOT_POINTS, PLOT_MULT
from image_server import ImageServer
from utils import setup_logging
from lidar_navigation.msg import Contour
from point2d import Point2D
from wall_finder import WallFinder


class LidarRansac(object):
    def __init__(self,
                 image_server=None,
                 iterations=20,
                 threshold=0.025,
                 min_points=20,
                 plot_all=False,
                 plot_centroid=False,
                 plot_points=False,
                 plot_mult=1.05,
                 contour_topic="/contour"):
        self.__iterations = iterations
        self.__threshold = threshold
        self.__min_points = min_points
        self.__plot_all = plot_all
        self.__plot_points = plot_points
        self.__plot_centroid = plot_centroid
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
                cnt = 0
                walls = []
                for w in WallFinder(iterations=self.__iterations,
                                    threshold=self.__threshold,
                                    min_points=self.__min_points,
                                    points=all_points).walls():
                    walls.append(w)
                    plt.plot([w.p0.x, w.p1.x], [w.p0.y, w.p1.y], 'r^', markersize=6.0)
                    # plt.plot([w.p0.x, w.p1.x], [w.p0.y, w.p1.y], 'b-')
                    plt.plot([p.x for p in w.points], [p.y for p in w.points], 'go', markersize=2.0)

                    m, b = w.slopeYInt()
                    if m > 1 or m < -1:
                        plt.plot([w.xfit(p.y) for p in w.points], [p.y for p in w.points], 'b-')
                    else:
                        plt.plot([p.x for p in w.points], [w.yfit(p.x) for p in w.points], 'r-')

                # if len(walls) != 3:
                print("Found {} walls {} {}".format(len(walls),
                                                    [len(w.points) for w in walls],
                                                    [w.slopeYInt()[0] for w in walls]))

            # Plot axis
            dist = max_dist * self.__plot_mult
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
                          cli.plot_centroid,
                          cli.plot_slices,
                          cli.plot_mult,
                          cli.contour_topic,
                          ImageServer.args,
                          cli.log_level)

    # Setup logging
    setup_logging(level=args[LOG_LEVEL])

    rospy.init_node('walls_node')

    image_server = ImageServer(template_file=args[TEMPLATE_FILE],
                               http_host=args[HTTP_HOST],
                               http_delay_secs=args[HTTP_DELAY_SECS],
                               http_verbose=args[HTTP_VERBOSE])

    image_server.start()
    image = LidarRansac(image_server=image_server,
                        plot_all=args[PLOT_ALL],
                        plot_points=args[PLOT_POINTS],
                        plot_centroid=args[PLOT_CENTROID],
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
