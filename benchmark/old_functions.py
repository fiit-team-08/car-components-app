import sys
sys.path.append(".")
import pandas as pd
import numpy as np
import math
import json
from sympy import Point, Segment
from scipy.spatial import distance
from similaritymeasures import curve_length_measure, frechet_dist
from obspy.geodetics import degrees2kilometers
from analysis.log_file_analyzer import drop_unnecessary_columns
from analysis.lap_difference_analyzer import shortest_distance


def analyze_laps(traces, reference_lap, laps):
    data_frame = pd.DataFrame(data={
        'pointsPerLap': [],
        'curveLength': [],
        'averagePerpendicularDistance': [],
        'lapData': []
    })

    for i in range(len(laps) - 1):
        lap_data = traces.iloc[laps[i]: laps[i + 1]]
        drop_unnecessary_columns(lap_data)
        # perpendicular_distance = find_out_difference_perpendiculars(lap_data, reference_lap)
        lap = {
            'pointsPerLap': len(lap_data),
            'curveLength': 0,
            'averagePerpendicularDistance': 0,
            'lapData': json.loads(lap_data.to_json(orient="records"))
        }
        data_frame = data_frame.append(lap, ignore_index=True)

    # tha last circuit (lap) was not saved yet so save that one
    lap_data = traces.iloc[laps[-1:]]
    drop_unnecessary_columns(lap_data)
    # perpendicular_distance = find_out_difference_perpendiculars(lap_data, reference_lap)
    lap = {
        'pointsPerLap': len(lap_data),
        'curveLength': 0,
        'averagePerpendicularDistance': 0,
        'lapData': json.loads(lap_data.to_json(orient="records"))
    }
    data_frame = data_frame.append(lap, ignore_index=True)
    return data_frame


def separate_laps(traces, ref_lap=None):
    """
        Separate all the log dataframe into several laps.

        Parameters
        --------
            ref_lap : DataFrame
                A dataframe with logs of a reference ride.
                It is used to define finish line.
                It is Optional parameter. Default value is None.
            traces : DataFrame
                A dataframe with logs of a ride.
            traces_id : int
                An ID of a ride. It is only used for naming of files.
            store_path : string
                A path where all the laps will be be stored.
    """

    ref_lap = traces if ref_lap is None else ref_lap
    points = list()
    for i in range(len(traces)):
        points.append([traces['LON'][i], traces['LAT'][i]])

    # use last points to determine normal vector
    last_point1 = [ref_lap['LON'].iloc[-1], ref_lap['LAT'].iloc[-1]]
    last_point2 = [ref_lap['LON'].iloc[-2], ref_lap['LAT'].iloc[-2]]

    a = last_point2[0] - last_point1[0]
    b = last_point2[1] - last_point1[1]

    dst = distance.euclidean(last_point1, last_point2)
    distance_multiplier = math.ceil(0.0001 / (2 * dst))

    v_normal = np.array([-b, a])
    start_point = np.array(last_point1)

    point_top = Point(start_point + distance_multiplier * v_normal, evaluate=False)
    point_bottom = Point(start_point - distance_multiplier * v_normal, evaluate=False)
    start_line = Segment(point_top, point_bottom, evaluate=False)

    laps = [0]

    for i in range(len(points) - 1):
        point1 = Point(points[i][0], points[i][1], evaluate=False)
        point2 = Point(points[i + 1][0], points[i + 1][1], evaluate=False)

        if point1 == point2:
            continue

        # segment between point1 and point2
        segment = Segment(point1, point2, evaluate=False)
        intersection = segment.intersection(start_line)

        # add start of a new lap
        if intersection:
            laps.append(i + 1)
            print('Lap ending at index: {}'.format(i))

    return laps


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


def line_length(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def create_vector(point_A, point_B):
    return [point_B.LAT - point_A.LAT, point_B.LON - point_A.LON]


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
