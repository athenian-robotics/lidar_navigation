import os

import sys

__path = os.path.abspath(sys.modules[__name__].__file__)
__dirname = os.path.dirname(__path)
HTTP_TEMPLATE_DEFAULT = __dirname + "/html/plot-image.html"
HTTP_PORT_DEFAULT = 8080
HTTP_HOST_DEFAULT = "localhost:{0}".format(HTTP_PORT_DEFAULT)
HTTP_DELAY_SECS_DEFAULT = 0

HTTP_HOST = "http_host"
HTTP_FILE = "http_file"
HTTP_DELAY_SECS = "http_delay_secs"
HTTP_VERBOSE = "http_verbose"
LOG_LEVEL = "loglevel"

SLICE_SIZE = "slice_size"
MAX_MULT = "max_mult"
MAX_MULT_DEFAULT = 1.1
SLICE_SIZE_DEFAULT = 5
PUBLISH_RATE = "publish_rate"
PUBLISH_RATE_DEFAULT = 30
SCAN_TOPIC = "scan_topic"
SCAN_TOPIC_DEFAULT = "/scan"
CONTOUR_TOPIC = "contour_topic"
CONTOUR_TOPIC_DEFAULT = "/contour"
CENTROID_TOPIC = "centroid_topic"
CENTROID_TOPIC_DEFAULT = "/centroid"
PC_TOPIC = "pc_topic"
PC_TOPIC_DEFAULT = "/pc2"

PLOT_MULT = "plot_mult"
PLOT_MULT_DEFAULT = 1.05

PLOT_ALL = "plot_all"
PLOT_CONTOUR = "plot_contour"
PLOT_POINTS = "plot_points"
PLOT_SLICES = "plot_slices"

PUBLISH_PC = "publish_pc"

MAX_LINEAR = "max_linear"
MAX_LINEAR_DEFAULT = .35
MAX_ANGULAR = "max_angular"
MAX_ANGULAR_DEFAULT = 2.75

VEL_TOPIC = "vel_topic"
VEL_TOPIC_DEFAULT = "/cmd_vel"

STOP_ANGLE = "stop_angle"
STOP_ANGLE_DEFAULT = 70
