import numpy as np
import pandas as pd
import math


WGS84 = (6378137, 298.257223563)


def geodetic_to_geocentric(lat: float, lon: float, ellps=None) -> (float, float):
    """
    Compute the Geocentric (Cartesian) Coordinates X, Y, Z
    given the Geodetic Coordinates lat, lon + Ellipsoid Height h
    """

    a, rf = ellps or WGS84
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    n = a / np.sqrt(1 - (1 - (1 - 1 / rf) ** 2) * (np.sin(lat_rad)) ** 2)
    x = n * np.cos(lat_rad) * np.cos(lon_rad)
    y = n * np.cos(lat_rad) * np.sin(lon_rad)

    return x, y


def find_closest_point(x1: float, y1: float, track: pd.DataFrame, last_index: int) -> int:
    min_index = last_index
    min_dist = np.inf

    for i in range(last_index, track.shape[0]):
        x2 = track['X'][i]
        y2 = track['Y'][i]
        dist = math.hypot(x2 - x1, y2 - y1)

        if dist < min_dist:
            min_index = i
            min_dist = dist

    return min_index
