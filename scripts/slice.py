import math

from point2d import Point2D


def to_point(rads, dist):
    x = math.cos(rads) * dist
    y = math.sin(rads) * dist
    return Point2D(x, y)


class Slice(object):
    def __init__(self, begin, end):
        self.__begin = begin
        self.__end = end

        self.__begin_rad = math.radians(self.__begin)
        self.__end_rad = math.radians(self.__end)

        self.__begin_point = None
        self.__end_point = None

        # Calculate the angle halfway between the begin and end
        self.__mid_rad = math.radians(self.__begin + ((self.__end - self.__begin) / 2.0))
        self.__nearest_point = None

    @property
    def nearest(self):
        return self.__nearest_point

    def __contains__(self, point):
        return self.__begin <= point.angle <= self.__end

    def begin_point(self, max_dist):
        return to_point(self.__begin_rad, max_dist)

    def end_point(self, max_dist):
        return to_point(self.__end_rad, max_dist)

    def add_point(self, point):
        # See if point is closer than the previously closest point
        if point.dist < self.__nearest_point.dist:
            self.__nearest_point = point

    def reset(self, max_dist):
        self.__nearest_point = to_point(self.__mid_rad, max_dist)

    def __str__(self):
        return "Begin: {} End: {} Nearest: {}".format(self.__begin, self.__end, self.__nearest_point)
