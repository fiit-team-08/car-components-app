import numpy as np
import pandas as pd
import math
import cv2


WGS84 = (6378137, 298.257223563)


def geodetic_to_geocentric(lat: float, lon: float) -> (float, float):
    """
    Compute the Geocentric (Cartesian) Coordinates X, Y
    given the Geodetic Coordinates lat, lon.
    """

    a, rf = WGS84
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    n = a / np.sqrt(1 - (1 - (1 - 1 / rf) ** 2) * (np.sin(lat_rad)) ** 2)
    x = n * np.cos(lat_rad) * np.cos(lon_rad)
    y = n * np.cos(lat_rad) * np.sin(lon_rad)

    return x, y


def find_closest_point(x1: float, y1: float, track: pd.DataFrame,
                       last_index: int) -> int:
    """
    Find the closest track point to the point given by x1, y1 coordinates.

    Parameters
    ----------
    x1 (float): X coordinate of the point.
    y1 (float): Y coordinate of the point.
    track (pd.DataFrame): DataFrame containing X and Y columns
        describing the track.
    last_index (int): Index from which the searching should start.

    Returns
    -------
    min_index (int): Index of the closest point from the track.
    """

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


def process_state_image(state: np.array) -> np.array:
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state


def generate_state_frame_stack_from_queue(deque) -> np.array:
    frame_stack = np.array(deque)
    # move stack dimension to the channel dimension
    # (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 0))
