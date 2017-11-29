import argparse
import logging

from constants import CENTROID_TOPIC, CENTROID_TOPIC_DEFAULT, PC_TOPIC, PC_TOPIC_DEFAULT
from constants import HTTP_DELAY_SECS, TEMPLATE_FILE, LOG_LEVEL, HTTP_HOST, HTTP_VERBOSE
from constants import HTTP_DELAY_SECS_DEFAULT, HTTP_HOST_DEFAULT, HTTP_TEMPLATE_DEFAULT
from constants import MAX_LINEAR, MAX_LINEAR_DEFAULT, MAX_ANGULAR, MAX_ANGULAR_DEFAULT
from constants import PAUSE, ITERATIONS, ITERATIONS_DEFAULT, MIN_POINTS, MIN_POINTS_DEFAULT
from constants import PLOT_ALL, PLOT_CONTOUR, PLOT_CENTROID, PLOT_POINTS, PLOT_SLICES
from constants import PUBLISH_RATE, PUBLISH_RATE_DEFAULT, MAX_AXIS_MULT, MAX_AXIS_MULT_DEFAULT
from constants import SCAN_TOPIC, SCAN_TOPIC_DEFAULT, CONTOUR_TOPIC, CONTOUR_TOPIC_DEFAULT
from constants import SLICE_SIZE, SLICE_SIZE_DEFAULT, SLICE_OFFSET, PUBLISH_PC, MAX_DIST_MULT, MAX_DIST_MULT_DEFAULT
from constants import THRESHOLD, THRESHOLD_DEFAULT
from constants import VEL_TOPIC, VEL_TOPIC_DEFAULT, STOP_ANGLE, STOP_ANGLE_DEFAULT


def setup_cli_args(*args):
    parser = argparse.ArgumentParser()
    for arg in args:
        if type(arg) is list:
            for a in arg:
                a(parser)
        else:
            arg(parser)
    return vars(parser.parse_args())


def http_host(p):
    return p.add_argument("--http", dest=HTTP_HOST, default=HTTP_HOST_DEFAULT,
                          help="HTTP hostname:port [{0}]".format(HTTP_HOST_DEFAULT))


def http_delay_secs(p):
    return p.add_argument("--delay", "--http_delay", dest=HTTP_DELAY_SECS, default=HTTP_DELAY_SECS_DEFAULT, type=float,
                          help="HTTP delay secs [{0}]".format(HTTP_DELAY_SECS_DEFAULT))


def template_file(p):
    return p.add_argument("--file", "--template_file", dest=TEMPLATE_FILE, default=HTTP_TEMPLATE_DEFAULT,
                          help="Template file name [{}]".format(HTTP_TEMPLATE_DEFAULT))


def http_verbose(p):
    return p.add_argument("--http_verbose", "--verbose_http", dest=HTTP_VERBOSE, default=False, action="store_true",
                          help="Enable verbose HTTP log [false]")


def log_level(p):
    return p.add_argument("-v", "--verbose", dest=LOG_LEVEL, default=logging.INFO, action="store_const",
                          const=logging.DEBUG, help="Enable debugging info")


def slice_size(p):
    return p.add_argument("--slice_size", dest=SLICE_SIZE, default=SLICE_SIZE_DEFAULT, type=int,
                          help="Slice size degrees [{0}]".format(SLICE_SIZE_DEFAULT))


def slice_offset(p):
    return p.add_argument("--slice_offset", dest=SLICE_OFFSET, default=0, type=int,
                          help="Slice offset [0]")


def max_dist_mult(p):
    return p.add_argument("--max_dist_mult", "--dist_mult", dest=MAX_DIST_MULT, default=MAX_DIST_MULT_DEFAULT,
                          type=float,
                          help="Maximum distance multiplier [{0}]".format(MAX_DIST_MULT_DEFAULT))


def max_axis_mult(p):
    return p.add_argument("--max_axis_mult", "--axis_mult", dest=MAX_AXIS_MULT, default=MAX_AXIS_MULT_DEFAULT,
                          type=float,
                          help="Maximum axis multiplier [{0}]".format(MAX_AXIS_MULT_DEFAULT))


def publish_rate(p):
    return p.add_argument("--publish_rate", dest=PUBLISH_RATE, default=PUBLISH_RATE_DEFAULT, type=int,
                          help="Publish rate [{0}]".format(PUBLISH_RATE_DEFAULT))


def scan_topic(p):
    return p.add_argument("--scan_topic", dest=SCAN_TOPIC, default=SCAN_TOPIC_DEFAULT,
                          help="LaserScan values topic name [{}]".format(SCAN_TOPIC_DEFAULT))


def contour_topic(p):
    return p.add_argument("--contour_topic", dest=CONTOUR_TOPIC, default=CONTOUR_TOPIC_DEFAULT,
                          help="Contour values topic name [{}]".format(CONTOUR_TOPIC_DEFAULT))


def centroid_topic(p):
    return p.add_argument("--centroid_topic", dest=CENTROID_TOPIC, default=CENTROID_TOPIC_DEFAULT,
                          help="Centroid values topic name [{}]".format(CENTROID_TOPIC_DEFAULT))


def vel_topic(p):
    return p.add_argument("--vel_topic", dest=VEL_TOPIC, default=VEL_TOPIC_DEFAULT,
                          help="Velocity values topic name [{}]".format(VEL_TOPIC_DEFAULT))


def pc_topic(p):
    return p.add_argument("--pc_topic", dest=PC_TOPIC, default=PC_TOPIC_DEFAULT,
                          help="PointCloud2 values topic name [{}]".format(PC_TOPIC_DEFAULT))


def publish_pc(p):
    return p.add_argument("--publish_pc", dest=PUBLISH_PC, default=False, action="store_true",
                          help="Publish point cloud data [false]")


def plot_all(p):
    return p.add_argument("--plot_all", "--all", dest=PLOT_ALL, default=False, action="store_true",
                          help="Plot all items [false]")

def plot_points(p):
    return p.add_argument("--plot_points", "--points", dest=PLOT_POINTS, default=False, action="store_true",
                          help="Plot point cloud [false]")


def plot_contour(p):
    return p.add_argument("--plot_contour", "--contour", dest=PLOT_CONTOUR, default=False, action="store_true",
                          help="Plot contour [false]")


def plot_centroid(p):
    return p.add_argument("--plot_centroid", "--centroid", dest=PLOT_CENTROID, default=False, action="store_true",
                          help="Plot centroid [false]")


def plot_slices(p):
    return p.add_argument("--plot_slices", "--slices", dest=PLOT_SLICES, default=False, action="store_true",
                          help="Plot slices [false]")


def max_linear(p):
    return p.add_argument("--max_linear", "--linear", dest=MAX_LINEAR, default=MAX_LINEAR_DEFAULT, type=float,
                          help="Maximum linear speed [{0}]".format(MAX_LINEAR_DEFAULT))


def max_angular(p):
    return p.add_argument("--max_angular", "--angular", dest=MAX_ANGULAR, default=MAX_ANGULAR_DEFAULT, type=float,
                          help="Maximum angular speed [{0}]".format(MAX_ANGULAR_DEFAULT))


def stop_angle(p):
    return p.add_argument("--stop_angle", dest=STOP_ANGLE, default=STOP_ANGLE_DEFAULT, type=int,
                          help="Linear stop angle [{0}]".format(STOP_ANGLE_DEFAULT))


def iterations(p):
    return p.add_argument("--iterations", "--iter", dest=ITERATIONS, default=ITERATIONS_DEFAULT, type=int,
                          help="Iterations per RANSEC point set [{}]".format(ITERATIONS_DEFAULT))


def min_num_points(p):
    return p.add_argument("--min_num_points", "--min_points", dest=MIN_POINTS, default=MIN_POINTS_DEFAULT, type=int,
                          help="Minimum number of points for a wall [{}]".format(MIN_POINTS_DEFAULT))


def threshold_distance(p):
    return p.add_argument("--threshold_distance", "--distance", dest=THRESHOLD, default=THRESHOLD_DEFAULT, type=float,
                          help="Threshold point distance [{}]".format(THRESHOLD_DEFAULT))


def pause(p):
    return p.add_argument("--pause", dest=PAUSE, default=0, type=int,
                          help="Pause secs per scan [0]")
