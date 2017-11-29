import logging
import time
from threading import Lock
from threading import Thread

import requests
import rospy
from flask import Flask
from flask import redirect
from flask import request
from werkzeug.wrappers import Response

import cli_args  as cli
from constants import HTTP_HOST_DEFAULT, HTTP_DELAY_SECS_DEFAULT, HTTP_PORT_DEFAULT

# Find where this package is installed
_image_fname = "/image.jpg"

logger = logging.getLogger(__name__)


class ImageServer(object):
    args = [cli.template_file, cli.http_host, cli.http_delay_secs, cli.http_verbose]

    def __init__(self,
                 template_file,
                 http_host=HTTP_HOST_DEFAULT,
                 http_delay_secs=HTTP_DELAY_SECS_DEFAULT,
                 http_verbose=False):
        self.__template_file = template_file
        self.__http_host = http_host
        self.__http_delay_secs = http_delay_secs

        vals = self.__http_host.split(":")
        self.__host = vals[0]
        self.__port = vals[1] if len(vals) == 2 else HTTP_PORT_DEFAULT

        self.__current_image_lock = Lock()
        self.__current_image = None
        self.__ready_to_stop = False
        self.__flask_launched = False
        self.__ready_to_serve = False
        self.__started = False
        self.__stopped = False

        if not http_verbose:
            class FlaskFilter(logging.Filter):
                def __init__(self, fname):
                    super(FlaskFilter, self).__init__()
                    self.__fname = "GET {0}".format(fname)

                def filter(self, record):
                    return self.__fname not in record.msg

            logging.getLogger('werkzeug').addFilter(FlaskFilter(_image_fname))

    @property
    def enabled(self):
        return len(self.__http_host) > 0

    @property
    def image(self):
        with self.__current_image_lock:
            if self.__current_image is None:
                return []
            # retval, buf = utils.encode_image(self.__current_image)
            return self.__current_image

    @image.setter
    def image(self, image):
        if not self.enabled:
            return

        # Wait until potential sleep in start() has completed
        if not self.__ready_to_serve:
            return

        if not self.__started:
            rospy.logerror("ImageServer.start() not called")
            return

        if not self.__flask_launched:
            height, width = 80 * 10, 80 * 10  # image.shape[:2]
            self._launch_flask(width, height)

        with self.__current_image_lock:
            self.__current_image = image

    def _launch_flask(self, width, height):
        flask = Flask(__name__)

        @flask.route('/')
        def index():
            return redirect("/image?delay={0}".format(self.__http_delay_secs))

        @flask.route('/image')
        def image_option():
            return get_page(request.args.get("delay"))

        @flask.route("/image" + "/<string:delay>")
        def image_path(delay):
            return get_page(delay)

        @flask.route(_image_fname)
        def image_jpg():
            response = Response(self.image, mimetype="image/jpeg")
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            return response

        @flask.route("/__shutdown__", methods=['POST'])
        def shutdown():
            if not self.__ready_to_stop:
                return "Not ready to stop"
            shutdown_func = request.environ.get('werkzeug.server.shutdown')
            if shutdown_func:
                self.__stopped = True
                shutdown_func()
            return "Shutting down..."

        def get_page(delay):
            delay_secs = int(delay) if delay else self.__http_delay_secs
            try:
                with open(self.__template_file) as f:
                    html = f.read()

                return html.replace("_TITLE_", "") \
                    .replace("_DELAY_SECS_", str(delay_secs)) \
                    .replace("_NAME_", "") \
                    .replace("_WIDTH_", str(width)) \
                    .replace("_HEIGHT_", str(height)) \
                    .replace("_IMAGE_FNAME_", _image_fname)
            except BaseException as e:
                rospy.logerror("Unable to create template file with %s [%s]", self.__template_file, e, exc_info=True)
                time.sleep(1)

        def run_http(flask_server, host, port):
            while not self.__stopped:
                try:
                    flask_server.run(host=host, port=port)
                except BaseException as e:
                    rospy.logerror("Restarting HTTP server [%s]", e, exc_info=True)
                    time.sleep(1)
                finally:
                    rospy.loginfo("HTTP server shutdown")

        # Run HTTP server in a thread
        Thread(target=run_http, kwargs={"flask_server": flask, "host": self.__host, "port": self.__port}).start()
        self.__flask_launched = True
        rospy.loginfo("Running HTTP server on http://%s:%s/", self.__host, self.__port)

    def _start(self):
        # Cannot start the flask server until the dimensions of the image are known
        # So do not fire up the thread until the first image is available
        rospy.loginfo("Using template file %s", self.__template_file)
        rospy.loginfo("Starting HTTP server on http://%s:%s/", self.__host, self.__port)
        self.__ready_to_serve = True
        self.__started = True

    def start(self):
        if self.__started:
            rospy.logerror("ImageServer.start() already called")
            return

        if self.__flask_launched or not self.enabled:
            return

        rospy.loginfo("Starting ImageServer")
        Thread(target=self._start).start()

    def stop(self):
        if not self.__flask_launched:
            return

        self.__ready_to_stop = True
        url = "http://{0}:{1}".format(self.__host, self.__port)
        rospy.loginfo("Shutting down %s", url)

        try:
            requests.post("{0}/__shutdown__".format(url))
        except requests.exceptions.ConnectionError:
            rospy.logerror("Unable to stop ImageServer")
