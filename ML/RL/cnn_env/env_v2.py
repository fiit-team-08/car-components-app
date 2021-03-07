# https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
import gym
from ML.RL.cnn_env import cnn
from gym import spaces
import pandas as pd
import numpy as np
import shapely.geometry as geom
from obspy.geodetics import degrees2kilometers
from ML.RL.bicycle_model import BicycleKinematicModel
from gym.envs.classic_control import rendering
from ML.RL.utils import find_closest_point

# 0.01 radians is about 6 degrees
# +/-0.05 is speed change

ACTIONS = np.array(
    [[0.0, 0.0], [0.0, 0.05], [0.0, -0.05],
     [0.05, 0.0], [0.05, 0.05], [0.05, -0.05],
     [-0.05, 0.0], [-0.05, 0.05], [-0.05, -0.05]]
)

WINDOW_W = 500
WINDOW_H = 500

SCALE = 1   # Track scale in the viewing window

CAR_WIDTH = 8.0
CAR_LENGTH = 20.0
CAR_COLOR = [1, 0, 0]
WINDOW_COLOR = [0, 0, 1]


# environment with discrete observation space
class CarEnv(gym.Env):
    """
    Environment class containing all the methods necessary for reinforcement
    learning.

    Attributes
    ----------
        df : pd.DataFrame
            DataFrame containing info about the ref. trace.
        road : geom.LineString
            Geometric multiline representing the ref. trace.
        observation_space : np.array
            Describes the set of all possible states by min and max values for
            each element of the attribute state. State/Observation consists of
            5 elements:
                x: x coord in meters
                y: y coord in meters
                theta: vehicle heading angle in radians
                v: velocity in m/s
                psi: steering angle in radians
        action_space : np.array
            Describes the set of all possible actions by min and max values
            for each element of the action array. Action consists of
            2 elements: velocity and psi.
        init_x : float
            Initial x position for reset.
        init_y : float
            Initial y position for reset.
        init_theta : float
            Initial vehicle heading angle for reset (radians).
        init_v : float
            Initial velocity of the vehicle for reset (m/s).
        init_psi : float
            Initial steering command of the vehicle given in radians.
        init_observation : np.array
            Initial observation used to reset the environment. It's described
            by array [x, y, theta, v, psi].
        state : np.array or None
            Current state of the car given by array [x, y, theta, v, psi].
        bicycle : BicycleKinematicModel or None
            Instance of BicycleKinematicModel for calculations of the new
            states.
        reward : float
            Negative reward accumulated with each step taken.
        last_point : index
            Index of last passed point on the ref trace.
        viewer : rendering.Viewer()
            Scene with objects being rendered
        track : list
            Coordinates x,y of all points of the track
        car_trans : rendering.Transformation()
            Attribute for making transformations with car object
        track_trans : rendering.Transformation()
            Attribute for making transformations with track
        left_wheel_trans, right_wheel_trans : rendering.Transformation()
            Attribute for making transformations with front wheels
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, filename='../../data/ref1.csv', type='discrete',
                 action_dim=2, verbose=1):

        super(CarEnv, self).__init__()
        self.type = type
        self.action_dim = action_dim
        self.verbose = verbose
        self.df = self._read_df(filename)
        self.road = self._create_road()

        # 130 km/h ~= 36 m/s
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, 0., -(np.pi / 4)]),
            high=np.array([np.inf, np.inf, np.pi, 36., np.pi / 4])
        )

        # steering cmd a.k.a. psi and velocity
        # delta speed limit per 0.1s - 10km/h in 2s
        self.action_space = spaces.Discrete(9)

        self.viewer = None
        self.init_x = self.df['LON'][0]
        self.init_y = self.df['LAT'][0]
        self.init_theta = self.df['CRS'][0]
        self.init_v = 2.0
        self.init_psi = 0.0
        self.init_observation = np.array([self.init_x, self.init_y,
                                          self.init_theta, self.init_v,
                                          self.init_psi])
        self.reward = 0.0
        self.last_point = None
        self.max_reached_point = 0
        self.state = None
        self.bicycle = None
        self.track = None
        self.car_trans = None
        self.left_wheel_trans = None
        self.right_wheel_trans = None
        self.left_rearwheel_trans = None
        self.right_rearwheel_trans = None
        self.track_trans = None

    @staticmethod
    def _read_df(filename: str) -> pd.DataFrame:
        df = pd.read_csv(filename)
        # covert degrees to meters and degrees to radians
        df['LAT'] = df['LAT'].apply(lambda deg: degrees2kilometers(deg) * 1000)
        df['LON'] = df['LON'].apply(lambda deg: degrees2kilometers(deg) * 1000)
        df['CRS'] = df['CRS'].apply(lambda deg: np.deg2rad(deg))

        return df

    def _create_road(self):
        self.points = list()

        for index, row in self.df.iterrows():
            self.points.append(geom.Point(row['LON'], row['LAT']))

        return geom.LineString(self.points)

    def _create_track(self):
        """
        Transforms world coordinates into x,y coordinates by finding center
        and translating points and creates track represented by x,y points

        :returns - list of x,y points representing track in
        """

        # gets border values of latitude and longitude
        min_lat = 5342900
        max_lat = 0
        min_lon = 1953500
        max_lon = 0
        for point in self.points:
            lon = point.x
            lat = point.y
            if lat > max_lat:
                max_lat = lat
            if lat < min_lat:
                min_lat = lat
            if lon > max_lon:
                max_lon = lon
            if lon < min_lon:
                min_lon = lon

        # gets center of map
        self.map_center = ((max_lon - min_lon) / 2 + min_lon,
                           (max_lat - min_lat) / 2 + min_lat)
        center_x, center_y = self.map_center

        # append all x,y points to the track list
        coordinates = []
        for point in self.points:
            x = point.x - center_x
            y = point.y - center_y
            coordinates.append((x * SCALE, y * SCALE))

        coordinates.append(((self.points[0].x - center_x) * SCALE,
                            (self.points[0].y - center_y) * SCALE))

        return coordinates

    def step(self, action):
        """
        Applies the action given as parameter and calculates the new state
        (observation).

        Parameters
        ----------
            action : np.array
                Action that should be performed.
                Consists of [delta_psi, delta_v], where
                delta_psi is change of the steering command in radians and
                delta_v is change of velocity in m/s.

        Returns
        ----------
            observation : np.array
                New state of the car.
            reward : float
                Reward for the performed action.
            done : bool
                True when the car moves further than 0.5m away from
                the ref. trace, otherwise false.
            info : dict
                Information about previous state, current state and reward.
        """
        # action = delta_psi in radians (delta of the steering command)
        # action = steering command a.k.a. angle psi in radians

        done = False
        info = {
            'current_state_x': self.state[0],
            'current_state_y': self.state[1],
            'current_state_theta': self.state[2],
            'current_state_v': self.state[3],
            'current_state_psi': self.state[4],
            'action': action
        }

        v = self.state[3] + action[1]
        delta_psi = action[0]

        action_repeat = 1
        total_reward = 0
        for i in range(action_repeat):
            self.bicycle.change_state(velocity=v, steering_rate=delta_psi)
            new_x, new_y, new_psi, new_theta = self.bicycle.get_state()
            new_point = geom.Point(new_x, new_y)
            distance = self.road.distance(new_point)

            self.reward -= 0.5
            reward = self.reward

            closest_point = find_closest_point(
                [new_x, new_y],
                self.df[['LAT', 'LON']].values.tolist(),
                self.last_point
            )
            if closest_point > self.max_reached_point:
                self.max_reached_point = closest_point

            point_diff = closest_point - self.last_point if self.last_point else 0.
            if point_diff < 0 and self.max_reached_point - closest_point > 100:
                reward += 100
                done = True
                break
            if point_diff < 0:
                reward -= 100

            self.last_point = closest_point

            reward += self.last_point * 2
            # exponential growth changes the range from 0-5 to 0-25
            # based on the distance from the ref trace
            reward -= ((0.5 - distance) * 10.0) ** 2
            total_reward += reward

            if distance > 0.5:
                total_reward -= 100
                done = True
                break

            # either the vehicle has reached the last point of the ref trace or it
            # has already come past that point, in which case the point_diff would
            # be negative value
            elif self.last_point == len(self.df):
                total_reward += 100
                done = True
                break

        self.state = [new_x, new_y, new_theta, v, new_psi]

        info['closest_point'] = self.last_point
        info['distance'] = distance
        info['reward'] = total_reward
        info['new_state_x'] = self.state[0]
        info['new_state_y'] = self.state[1]
        info['new_state_theta'] = self.state[2]
        info['new_state_v'] = self.state[3]
        info['new_state_psi'] = self.state[4]

        # update info (include previous state, new state and reward)
        observation = self.state

        # return observation, reward, done, info
        return observation, total_reward, done, info

    def reset(self):
        """
        Resets the state of the car to the initial state.
        """
        self.state = self.init_observation
        self.bicycle = BicycleKinematicModel(x=self.state[0],
                                             y=self.state[1],
                                             heading_angle=self.state[2],
                                             steering_angle=self.state[4]
                                             )
        self.reward = 0.0
        self.last_point = None
        self.max_reached_point = 0
        # assigns points of track
        self.track = self._create_track()

        return self.init_observation

    def render(self, mode='human'):
        """
        Renders screen window with road represented by line and actual position
        of car object.

        :return - viewer with drawn objects
        """

        x, y, theta, v, psi = self.state
        center_x, center_y = self.map_center
        x = x - center_x
        y = y - center_y

        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            left, right, top, bottom = (
                -CAR_WIDTH / 2,
                CAR_WIDTH / 2,
                CAR_LENGTH / 2,
                -CAR_LENGTH / 2
            )

            # CAR BODY
            car = rendering.FilledPolygon([
                (left, bottom),
                (left, top),
                (right, top),
                (right, bottom)
            ])
            self.car_trans = rendering.Transform()
            car.add_attr(self.car_trans)
            car.set_color(1.0, .0, .0)

            # FRONT WINDOW
            # window_front = rendering.FilledPolygon([
            #     (left + 2, bottom + 20),
            #     (left + 2, top - 10),
            #     (right - 2, top - 10),
            #     (right - 2, bottom + 20)
            # ])
            # window_front.add_attr(self.car_trans)
            # window_front.set_color(.5, .5, .8)

            # REAR WINDOW
            # window_rear = rendering.FilledPolygon([
            #     (left + 2, bottom + 5),
            #     (left + 2, top - 30),
            #     (right - 2, top - 30),
            #     (right - 2, bottom + 5)
            # ])
            # window_rear.add_attr(self.car_trans)
            # window_rear.set_color(.5, .5, .8)

            # FRONT WHEELS
            left_front_wheel = rendering.FilledPolygon([
                (left - 2, bottom + 30),
                (left - 2, top - 1),
                (right - 15, top - 1),
                (right - 15, bottom + 30)
            ])
            self.left_wheel_trans = rendering.Transform()
            left_front_wheel.add_attr(self.left_wheel_trans)
            left_front_wheel.set_color(.1, .2, .1)

            right_front_wheel = rendering.FilledPolygon([
                (left + 15, bottom + 30),
                (left + 15, top - 1),
                (right + 2, top - 1),
                (right + 2, bottom + 30)
            ])
            self.right_wheel_trans = rendering.Transform()
            right_front_wheel.add_attr(self.right_wheel_trans)
            right_front_wheel.set_color(.1, .2, .1)

            # REAR WHEELS
            left_rear_wheel = rendering.FilledPolygon([
                (left - 2, bottom + 1),
                (left - 2, top - 30),
                (right - 15, top - 30),
                (right - 15, bottom + 1)
            ])
            left_rear_wheel.add_attr(self.car_trans)
            left_rear_wheel.set_color(.1, .2, .1)

            right_rear_wheel = rendering.FilledPolygon([
                (left + 15, bottom + 1),
                (left + 15, top - 30),
                (right + 2, top - 30),
                (right + 2, bottom + 1)
            ])
            right_rear_wheel.add_attr(self.car_trans)
            right_rear_wheel.set_color(.1, .2, .1)

            # centers points of the track to the center of the screen
            for i in range(len(self.track)):
                x, y = self.track[i]
                self.track[i] = x + WINDOW_W/2, y + WINDOW_H/2
                x, y = self.track[i]

            # creates track represented by line
            track = rendering.PolyLine(self.track, 0)
            self.track_trans = rendering.Transform()
            track.add_attr(self.track_trans)
            self.viewer.add_geom(track)

            self.viewer.add_geom(car)
            # self.viewer.add_geom(window_front)
            # self.viewer.add_geom(window_rear)
            self.viewer.add_geom(left_front_wheel)
            self.viewer.add_geom(right_front_wheel)
            self.viewer.add_geom(left_rear_wheel)
            self.viewer.add_geom(right_rear_wheel)

        if self.state is None:
            return None

        # Transforms objects with each rendering
        self.car_trans.set_translation(x + WINDOW_W/2, y + WINDOW_H/2)
        self.car_trans.set_rotation(-theta)

        self.left_wheel_trans.set_translation(x + WINDOW_W / 2,
                                              y + WINDOW_H / 2)
        self.right_wheel_trans.set_translation(x + WINDOW_W / 2,
                                               y + WINDOW_H / 2)

        psi = np.pi / 4 if psi > np.pi / 4 else psi
        psi = - np.pi / 4 if psi < - np.pi / 4 else psi
        self.left_wheel_trans.set_rotation(-theta - psi)
        self.right_wheel_trans.set_rotation(-theta - psi)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """
        Closes the viewer window.
        """

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def get_image_slice(image, env):
    slice_size = 96
    x = env.state[0] - env.map_center[0] + WINDOW_W / 2
    y = env.state[1] - env.map_center[1] + WINDOW_H / 2

    x1 = np.floor(x - slice_size // 2)
    x1 = int(x1)
    x2 = np.floor(x + slice_size // 2)
    x2 = int(x2)

    y1 = WINDOW_H - np.floor(y + slice_size // 2)
    y1 = int(y1)
    y2 = WINDOW_H - np.floor(y - slice_size // 2)
    y2 = int(y2)

    return image[x1:x2, y1:y2]


if __name__ == '__main__':
    agent = cnn.Agent()
    env = CarEnv()
    running_score = 0
    for i_episode in range(20000):
        score = 0
        observation = env.reset()

        for t in range(10000):
            rgb_array = env.render('rgb_array')

            slice = get_image_slice(rgb_array, env)
            gray_array = cnn.Agent.rgb2gray(slice)
            state = np.array([gray_array] * cnn.img_stack)

            action, a_logp = agent.select_action(state)
            env_action = action * np.array([2., 2.]) + np.array([-1., -1])         # That can make action having negatives values

            observation, reward, done, info = env.step(env_action)
            score += reward

            state_ = env.render('rgb_array')

            state_ = get_image_slice(state_, env)
            state_ = cnn.Agent.rgb2gray(state_)
            state_ = np.array([state_] * cnn.img_stack)

            if agent.store((state, action, a_logp, reward, state_)):
                print('Optimizing...')
                agent.update()

            print(info)
            if done:
                break

        running_score = running_score * 0.99 + score * 0.01
        print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_episode, score, running_score))

    agent.save_param()
    env.close()

