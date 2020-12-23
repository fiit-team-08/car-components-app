import math
from math import sqrt


def line_length(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_closest_point(point, lap, last_index=None):
    min_index = 0
    min_length = math.inf
    index_range = (
        range(len(lap)) if not last_index else range(max(0, last_index - 10), len(lap))
    )

    for i in index_range:
        lat = lap[i][0]
        lon = lap[i][1]

        length = line_length(lon, lat, point[0], point[1])
        if length < min_length:
            min_index = i
            min_length = length

    return min_index
