import pandas as pd
from pandas import DataFrame
import numpy as np
import math
from sympy import Point, Segment
from scipy.spatial import distance


def log_to_dataFrame(file_path):
    """
        Converts a log file of a ride to a Pandas dataframe.

        Parameters
        --------
            file_path : str
                A path to a log file.

        Example of a log file
        --------
            2020-06-29 13:06:24,595 - INFO - ;LAT;480492306;LON;175678507;UTMX;69136106;UTMY;532496222;HMSL;126112;GSPEED;0;CRS;0;HACC;66720;NXPT;1139
            2020-06-29 13:06:24,648 - INFO - ;LAT;480492313;LON;175678494;UTMX;69136096;UTMY;532496230;HMSL;126121;GSPEED;4;CRS;0;HACC;52510;NXPT;1139
            2020-06-29 13:06:24,698 - INFO - ;LAT;480492305;LON;175678495;UTMX;69136097;UTMY;532496221;HMSL;126146;GSPEED;1;CRS;0;HACC;49421;NXPT;1140

        Returns
        --------
        A dataframe with all the logs. 
    """

    logs = pd.read_csv(file_path, header=None, sep=';', names=['0', '1', 'LAT', '3', 'LON', '5', 'UTMX', '7', 'UTMY',
                                                               '9', 'HMSL', '11', 'GSPEED', '13', 'CRS', '15', 'HACC',
                                                               '17', 'NXPT'])

    logs = logs.drop(columns=['0', '1', '3', '5', '7', '9', '11', '13', '15', '17'])
    logs = logs.dropna()
    return logs


def normalize_logs(logs):
    """
        Normalizes data of the logs dataframe.
        In particular, the 'LAT' and 'LON' columns is divided by 10 000 000.
        The 'GSPEED' column is divided by 100.
        The CRS column is divided by 100 000.

        Parameters
        --------
            logs : DataFrame
                A dataframe with logs of a ride.
    """

    logs['LAT'] = logs['LAT'].apply(lambda x: x * 0.0000001)
    logs['LON'] = logs['LON'].apply(lambda x: x * 0.0000001)
    logs['GSPEED'] = logs['GSPEED'].apply(lambda x: x * 0.01)
    logs['CRS'] = logs['CRS'].apply(lambda x: x * 0.00001)


def drop_unnecessary_columns(logs):
    """
        Drops the columns 'UTMX', 'UTMY', 'HMSL', 'HACC' and 'NXPT' of the logs dataframe.

        Parameters
        --------
            logs : DataFrame
                A dataframe with logs of a ride.
    """

    logs.drop(columns=['UTMX', 'UTMY', 'HMSL', 'HACC', 'NXPT'], inplace=True)


def drop_logs_where_car_stayed(logs: DataFrame):
    """
        Drops rows from the logs dataframe where the LAT and LON are not changing.
        Resets indices of a dataframe in the end.

        Parameters
        --------
            logs : DataFrame
                A dataframe with logs of a ride.
    """

    last_lat = None
    last_lon = None
    dropped_rows = list()

    for index, row in logs.iterrows():
        if row['LAT'] == last_lat and row['LON'] == last_lon:
            dropped_rows.append(index)
        else:
            last_lat = row['LAT']
            last_lon = row['LON']

    logs.drop(dropped_rows, inplace=True)
    logs.reset_index(drop=True, inplace=True)


def create_columns_with_future_position(logs):
    """
        Creates columns NLAT, NLON and NCRS which are the next position of a car.

        Parameters
        --------
            logs : DataFrame
                A dataframe with logs of a ride.
    """

    next_lat = logs['LAT']
    next_lat = next_lat.append(pd.Series([np.nan]), ignore_index=True)
    next_lat = next_lat.iloc[1:]
    next_lat = next_lat.reset_index(drop=True)

    next_lon = logs['LON']
    next_lon = next_lon.append(pd.Series([np.nan]), ignore_index=True)
    next_lon = next_lon.iloc[1:]
    next_lon = next_lon.reset_index(drop=True)

    next_crs = logs['CRS']
    next_crs = next_crs.append(pd.Series([np.nan]), ignore_index=True)
    next_crs = next_crs.iloc[1:]
    next_crs = next_crs.reset_index(drop=True)

    logs['NLAT'] = next_lat
    logs['NLON'] = next_lon
    logs['NCRS'] = next_crs

    logs = logs.dropna()  # Drop the last row which contains NaN values.


def separate_laps(traces, traces_id, store_path, ref_lap = None):
    """
        Separate all the log dataframe into several laps.
        In the end laps are stored to files. 

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

    ref_lap = traces if ref_lap == None else ref_lap
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

    # save the circuits (laps)
    for i in range(len(laps) - 1):
        lap_df = traces.iloc[laps[i]: laps[i + 1]]
        lap_df.to_csv('{}/lap{}-{}.csv'.format(store_path, traces_id, i), index=False)

    # tha last circuit (lap) was not saved yet so save that one
    lap_df = traces.iloc[laps[-1]:]
    lap_df.to_csv('{}/lap{}-{}.csv'.format(store_path, traces_id, len(laps) - 1), index=False)


def normalize_for_graph(logs):
    """
        Drops all columns except LAT, and LON

        Parameters
        --------
            logs : DataFrame
                A dataframe with logs of a ride.
    """
    logs.drop(columns=['UTMX', 'UTMY', 'HMSL', 'GSPEED', 'CRS', 'HACC', 'NXPT'], inplace=True)
    logs.rename(columns={"LAT": "y", "LON": "x"}, inplace=True)


def get_raw_data(file_path) -> DataFrame:
    log_df = log_to_dataFrame(file_path)
    normalize_logs(log_df)
    return log_df


def get_essential_data(file_path) -> DataFrame:
    log_df = log_to_dataFrame(file_path)
    normalize_logs(log_df)
    drop_unnecessary_columns(log_df)
    drop_logs_where_car_stayed(log_df)
    return log_df


def get_graph_data(file_path) -> DataFrame:
    log_df = log_to_dataFrame(file_path)
    normalize_logs(log_df)
    normalize_for_graph(log_df)
    return log_df


def get_raw_data_json(file_path) -> str:
    data = get_raw_data(file_path)
    return data.to_json(orient="records")


def get_essential_data_json(file_path) -> str:
    data = get_essential_data(file_path)
    return data.to_json(orient="records")


def get_track_graph_data(file_path) -> str:
    data = get_graph_data(file_path)
    return data.to_json(orient="records")
