import shapely.geometry
from analysis.log_file_analyzer import *
from math import atan2
from command_prediction.bicycle import BicycleKinematicModel
from shapely.geometry import LineString
from shapely.geometry.polygon import LinearRing


def angle_between(v1, v2, positive=False):
    """
        Computes an angle between two vectors
    """
    angle = atan2(np.linalg.det(np.array(v1 + v2).reshape((2, 2))), np.dot(v1, v2))
    if positive and angle < 0:
        angle += 2 * math.pi
    return angle


def get_new_steering_angles(lap, reference_lap, speed=1):
    """
        Computes commands using bicycle model in each step of ride to keep the vehicle on track
        Parameters
        --------
            lap : DataFrame
                A dataframe with logs of a ride.
            reference_lap : DataFrame
                A dataframe with reference ride.
            speed : Float
                Constant speed of the vehicle, default is 1.
    """

    points = [tuple(x) for x in lap[['LON', 'LAT']].to_numpy()]
    reference_points = [tuple(x) for x in reference_lap[['LON', 'LAT']].to_numpy()]
    reference_polygon = LinearRing(reference_lap[['LON', 'LAT']].values)

    possible_angles = []
    for i in range(30, 0, -1):
        possible_angles.append(i)
    for i in range(1, 30):
        possible_angles.append(-i)

    # use last points to determine normal vector
    last_point1 = [reference_lap['LON'].iloc[-1], reference_lap['LAT'].iloc[-1]]
    last_point2 = [reference_lap['LON'].iloc[-2], reference_lap['LAT'].iloc[-2]]

    a = last_point2[0] - last_point1[0]
    b = last_point2[1] - last_point1[1]

    dst = distance.euclidean(last_point1, last_point2)
    distance_multiplier = math.ceil(10 / (2 * dst))

    v_normal = np.array([-b, a])
    start_point = np.array(last_point1)

    point_top = start_point + distance_multiplier * v_normal
    point_bottom = start_point - distance_multiplier * v_normal
    start_segment = segment(point_top, point_bottom)

    starting_point = points[0]
    x, y = starting_point
    xt, yt = reference_points[0]
    nearest_vector = (xt - x, yt - y)
    starting_heading_angle = angle_between((1, 0), nearest_vector, True)
    model = BicycleKinematicModel(x=starting_point[0],
                                  y=starting_point[1],
                                  heading_angle=starting_heading_angle,
                                  steering_angle=0,
                                  time_step=1)
    actual_index = 0
    prev_point = None

    created_points = []
    angles = []
    steering_angles = []
    heading_angles = []

    while True:
        actual_state = model.get_state()
        current_x, current_y, steering_angle, heading_angle = actual_state
        created_points.append((current_x, current_y))
        steering_angles.append(steering_angle)
        heading_angles.append(heading_angle)

        actual_point = [current_x, current_y]
        if prev_point is not None:
            if intersection(segment(prev_point, actual_point), start_segment):
                print("Lap has {} points".format(actual_index))
                break

        prev_point = actual_point

        turn_angle = 0
        lowest_distance = math.inf
        for j in range(len(possible_angles)):
            copied_model = BicycleKinematicModel(x=current_x,
                                                 y=current_y,
                                                 steering_angle=steering_angle,
                                                 heading_angle=heading_angle,
                                                 time_step=1)
            angle = math.radians(possible_angles[j])

            copied_model.change_state(speed, angle)
            copied_model.change_state(speed, angle)
            copied_model.change_state(speed, angle)

            new_x, new_y, new_steering_angle, _ = copied_model.get_state()
            new_distance = shapely.geometry.Point([new_x, new_y]).distance(reference_polygon)
            line = LineString([[current_x, current_y], [new_x, new_y]])
            is_near_reference = reference_polygon.intersects(line)
            if lowest_distance > new_distance and is_near_reference and math.radians(
                    30) > new_steering_angle > math.radians(-30):
                lowest_distance = new_distance
                turn_angle = angle

        model.change_state(speed, turn_angle)
        angles.append(turn_angle)
        actual_index += 1

    return created_points, angles, steering_angles, heading_angles


def get_simple_command_prediction(reference_lap_file, traces_file, speed=1):
    """
        Creates DataFrames from both files and computes probable steering angles and returns a new DataFrame containing new data
        Parameters
        --------
            reference_lap_file : String
                path to reference ride logs.
            traces_file : String
                path to traces logs.
            speed : Float
                Speed of vehicle.
    """
    reference_df = log_to_dataFrame(reference_lap_file)
    normalize_logs(reference_df)

    traces_df = log_to_dataFrame(traces_file)
    normalize_logs(traces_df)

    reference_df['LAT'] = reference_df['LAT'].apply(lambda deg: degrees2kilometers(deg) * 1000)
    reference_df['LON'] = reference_df['LON'].apply(lambda deg: degrees2kilometers(deg) * 1000)
    reference_df['CRS'] = reference_df['CRS'].apply(lambda deg: np.deg2rad(deg))

    traces_df['LAT'] = traces_df['LAT'].apply(lambda deg: degrees2kilometers(deg) * 1000)
    traces_df['LON'] = traces_df['LON'].apply(lambda deg: degrees2kilometers(deg) * 1000)
    traces_df['CRS'] = traces_df['CRS'].apply(lambda deg: np.deg2rad(deg))

    lap = traces_df.iloc[0:2]

    created_points, angles, steering_angles, heading_angles = get_new_steering_angles(lap, reference_df, speed)
    list_x, list_y = list(map(list, zip(*created_points)))
    d = {
        'x': list_x,
        'y': list_y,
        'CRS': heading_angles,
        'TIME': np.linspace(reference_df.iloc[0]['TIME'], reference_df.iloc[-1]['TIME'], len(heading_angles))
    }
    df = pd.DataFrame(data=d)
    df.x -= df.x[0]
    df.y -= df.y[0]
    df['CRS'] = df['CRS'].apply(lambda deg: np.rad2deg(deg) % 360)
    # minus = False
    # for i in range(df.CRS.size - 1):
    #     if (df.CRS[i] - df.CRS[i+1]) > 300:
    #         minus = False
    #         temp = df.CRS[i]
    #         df.CRS[i] = -abs(360 - temp)
    #     if minus:
    #         temp = df.CRS[i]
    #         df.CRS[i] = -abs(360 - temp)
    #     if (df.CRS[i+1] - df.CRS[i]) > 300:
    #         minus = True
    return df



def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y
