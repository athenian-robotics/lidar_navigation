import math

from geometry_msgs.msg import Point


class Point2D(object):
    def __init__(self, x, y):
        self.__x = float(x)
        self.__y = float(y)
        self.__dist = math.sqrt(self.__x ** 2 + self.__y ** 2)

        if self.__x == 0:
            self.__angle = 90.0
        else:
            degrees = math.degrees(math.atan(self.__y / self.__x))
            self.__angle = degrees if self.__x > 0 else 90 + (90 - abs(degrees))

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def dist(self):
        return self.__dist

    @property
    def angle(self):
        return self.__angle

    @property
    def heading(self):
        # Returns negative for counterclockwise and positive for clockwise
        return round(90 - self.angle, 2)

    def to_ros_point(self):
        return Point(self.x, self.y, 0.0)

    def __str__(self):
        return "({}, {}) Dist: {} Angle: {} Heading: {}".format(self.x, self.y, self.dist, self.angle, self.heading)


Origin = Point2D(0, 0)
