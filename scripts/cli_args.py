import argparse
import logging

from constants import CENTROID_TOPIC, CENTROID_TOPIC_DEFAULT, PC_TOPIC, PC_TOPIC_DEFAULT
from constants import HTTP_DELAY_SECS, HTTP_FILE, LOG_LEVEL, HTTP_HOST, HTTP_VERBOSE
from constants import HTTP_DELAY_SECS_DEFAULT, HTTP_HOST_DEFAULT, HTTP_TEMPLATE_DEFAULT
from constants import PLOT_ALL, PLOT_CONTOUR, PLOT_POINTS, PLOT_SLICES
from constants import PUBLISH_RATE, PUBLISH_RATE_DEFAULT, PLOT_MULT, PLOT_MULT_DEFAULT
from constants import SCAN_TOPIC, SCAN_TOPIC_DEFAULT, CONTOUR_TOPIC, CONTOUR_TOPIC_DEFAULT
from constants import SLICE_SIZE, SLICE_SIZE_DEFAULT, PUBLISH_PC, MAX_MULT, MAX_MULT_DEFAULT


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


def http_file(p):
    return p.add_argument("-i", "--file", "--http_file", dest=HTTP_FILE, default=HTTP_TEMPLATE_DEFAULT,
                          help="HTTP template file [{}]".format(HTTP_TEMPLATE_DEFAULT))


def http_verbose(p):
    return p.add_argument("--http_verbose", "--verbose_http", dest=HTTP_VERBOSE, default=False, action="store_true",
                          help="Enable verbose HTTP log [false]")


def log_level(p):
    return p.add_argument("-v", "--verbose", dest=LOG_LEVEL, default=logging.INFO, action="store_const",
                          const=logging.DEBUG, help="Enable debugging info")


def slice_size(p):
    return p.add_argument("--slice_size", dest=SLICE_SIZE, default=SLICE_SIZE_DEFAULT, type=int,
                          help="Slice size degrees [{0}]".format(SLICE_SIZE_DEFAULT))


def max_mult(p):
    return p.add_argument("--max_mult", dest=MAX_MULT, default=MAX_MULT_DEFAULT, type=float,
                          help="Maximum distance multiplier [{0}]".format(MAX_MULT_DEFAULT))


def plot_mult(p):
    return p.add_argument("--plot_mult", dest=PLOT_MULT, default=PLOT_MULT_DEFAULT, type=float,
                          help="Maximum plot multiplier [{0}]".format(PLOT_MULT_DEFAULT))


def publish_rate(p):
    return p.add_argument("--publish_rate", dest=PUBLISH_RATE, default=PUBLISH_RATE_DEFAULT, type=int,
                          help="Publish rate [{0}]".format(PUBLISH_RATE_DEFAULT))


def scan_topic(p):
    return p.add_argument("--scan_topic", dest=SCAN_TOPIC, default=SCAN_TOPIC_DEFAULT,
                          help="Scan topic name [{}]".format(SCAN_TOPIC_DEFAULT))


def contour_topic(p):
    return p.add_argument("--contour_topic", dest=CONTOUR_TOPIC, default=CONTOUR_TOPIC_DEFAULT,
                          help="Contour topic name [{}]".format(CONTOUR_TOPIC_DEFAULT))


def centroid_topic(p):
    return p.add_argument("--centroid_topic", dest=CENTROID_TOPIC, default=CENTROID_TOPIC_DEFAULT,
                          help="Centroid topic name [{}]".format(CENTROID_TOPIC_DEFAULT))


def pc_topic(p):
    return p.add_argument("--pc_topic", dest=PC_TOPIC, default=PC_TOPIC_DEFAULT,
                          help="Point cloud topic anme [{}]".format(PC_TOPIC_DEFAULT))


def publish_pc(p):
    return p.add_argument("--publish_pc", dest=PUBLISH_PC, default=False, action="store_true",
                          help="Publish point cloud data [false]")


def plot_all(p):
    return p.add_argument("--plot_all", "--all", dest=PLOT_ALL, default=False, action="store_true",
                          help="Plot all items [false]")

def plot_contour(p):
    return p.add_argument("--plot_contour", "--contour", dest=PLOT_CONTOUR, default=False, action="store_true",
                          help="Plot contour [false]")


def plot_points(p):
    return p.add_argument("--plot_points", "--points", dest=PLOT_POINTS, default=False, action="store_true",
                          help="Plot point cloud [false]")


def plot_slices(p):
    return p.add_argument("--plot_slices", "--slices", dest=PLOT_SLICES, default=False, action="store_true",
                          help="Plot slices [false]")
