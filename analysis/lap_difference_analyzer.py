import numpy as np
import pandas as pd
import math
from numpy.linalg import norm
from similaritymeasures import curve_length_measure, frechet_dist
from obspy.geodetics import degrees2kilometers


def create_curve(dataframe):
    curve = np.zeros((dataframe.shape[0], 2))
    curve[:, 0] = dataframe.LON
    curve[:, 1] = dataframe.LAT
    return curve


def find_out_difference(ref_lap, laps):
    """
        With the usage of several curve metrics, finds out differences between 
        a referrence lap and laps of a ride 

        Parameters
        --------
            ref_lap : DataFrame
                A dataframe of with logs of a a reference ride.
            laps : list
                A list of dataframes. 
                Each dataframe represents one lap of a ride.
        
        Returns
        --------
            A dataframe object with three columns: a Measurements count, a Frechet distance and a Curve length measure.
    """

    ref_curve = create_curve(ref_lap)

    measurement_column = 'Measurements count'
    frechet_column = 'Frechet distance'
    curve_len_column = 'Curve length measure'
    data_structure = {measurement_column: [],
                      frechet_column: [],
                      curve_len_column: []}

    differences_df = pd.DataFrame(data=data_structure)

    for lap in laps:
        experimental_curve = create_curve(lap)

        m_count = len(lap)
        fd = frechet_dist(experimental_curve, ref_curve)
        cl = curve_length_measure(experimental_curve, ref_curve)

        difference = {measurement_column: m_count,
                      frechet_column: fd,
                      curve_len_column: cl, }

        differences_df = differences_df.append(difference, ignore_index=True)

    return differences_df


def line_length(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_closest_point(point, lap):
    minIndex = 0
    minLength = math.inf
    for i in lap.index:
        lat = lap.loc[i].LAT
        lon = lap.loc[i].LON

        length = line_length(lat, lon, point.LAT, point.LON)
        if length < minLength:
            minIndex = i
            minLength = length

    return minIndex


def find_angle_between_vectors(vector_A, vector_B):
    unit_vector_A = vector_A / np.linalg.norm(vector_A)
    unit_vector_B = vector_B / np.linalg.norm(vector_B)
    dot_product = np.dot(unit_vector_A, unit_vector_B)
    return np.arccos(dot_product)  # return angle in radians


def create_vector(point_A, point_B):
    return [point_B.LAT - point_A.LAT, point_B.LON - point_A.LON]


# Perdendicular from p1 to line (p2,p3)
def shortest_distance(p1, p2, p3):
    dist = norm(np.cross(p2 - p3, p3 - p1)) / norm(p3 - p2)
    return dist


def find_shortest_distance(p1, p2, p3):
    x = np.array([p1.LAT, p1.LON])
    y = np.array([p2.LAT, p2.LON])
    z = np.array([p3.LAT, p3.LON])
    return shortest_distance(x, y, z)


def find_out_difference_perpendiculars(lap, ref_lap):
    """
        Calculates perpendiculars from every point of a lap to a ref_lap.

        Parameters
        --------
            lap : DataFrame
                A dataframe with a lap from which perpendiculars are calculated. 
            ref_lap : DataFrame
                A dataframe with a lap to which perpenduculars are calculated.
        
        Returns
        --------
            A list of perpendiculars from lap to ref_lap.
    """

    distances = []

    for i in lap.index:
        point = lap.loc[i]

        closest_index = find_closest_point(point, ref_lap)
        closest_point = ref_lap.loc[closest_index]

        neighbor_i = len(ref_lap) - 1 if closest_index == 0 else closest_index - 1
        neighbor1 = ref_lap.loc[neighbor_i]
        neighbor_i = 0 if len(ref_lap) == closest_index + 1 else closest_index + 1
        neighbor2 = ref_lap.loc[neighbor_i]

        v1 = create_vector(closest_point, point)
        v2 = create_vector(closest_point, neighbor1)
        v3 = create_vector(closest_point, neighbor2)

        angle1 = find_angle_between_vectors(v1, v2)
        angle2 = find_angle_between_vectors(v1, v3)

        degrees90 = math.pi / 2
        min_dist = -1
        if angle1 > degrees90 and angle2 > degrees90:
            min_dist = line_length(point.LAT, point.LON, closest_point.LAT, closest_point.LON)
        elif angle1 < degrees90 and angle2 < degrees90:
            dist1 = find_shortest_distance(point, closest_point, neighbor1)
            dist2 = find_shortest_distance(point, closest_point, neighbor2)
            min_dist = dist1 if dist1 <= dist2 else dist2
        elif angle1 <= degrees90:
            min_dist = find_shortest_distance(point, closest_point, neighbor1)
        elif angle2 <= degrees90:
            min_dist = find_shortest_distance(point, closest_point, neighbor2)

        if min_dist == -1:
            print('ERROR: Could not find distance')
            print("Indices: {} {}\nAngles: {} {}".format(i, closest_index, angle1, angle2))
        elif math.isnan(min_dist):
            print("NAN value!!!\nIndices: {} {}\nAngles: {} {}".format(i, closest_index, angle1, angle2))
        elif min_dist < 0:
            print("Negative value!!!\nIndices: {} {}\nAngles: {} {}".format(i, closest_index, angle1, angle2))
        else:
            min_dist = degrees2kilometers(min_dist) * 100000  # in centimeters
            distances.append(min_dist)

    return distances
