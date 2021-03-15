#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import geopandas
import random
from gym import Env, spaces
from shapely.geometry import Polygon, Point
from ML.RL.car_env.simulator import Simulator
from ML.RL.car_env.utils import geodetic_to_geocentric
from ML.RL.car_env.utils import find_closest_point


class CarEnv(Env):
    def __init__(self, type='continuous', action_dim=2, actions=None, tolerance=1.0, filename='../../data/ref1.csv'):
        # tolerance in meters
        self.type = type
        self.action_dim = action_dim
        self.tolerance = tolerance

        self.df = self._read_df(filename)
        self.track = self._create_track()

        # steering cmd a.k.a. psi and
        # 0.005 rad ~= 3 deg
        # 0.1 m/s per
        self.actions = actions or np.array(
            [[0.0, 0.0], [0.0, 0.1], [0.0, -0.1],
             [0.005, 0.0], [0.005, 0.1], [0.005, -0.1],
             [-0.005, 0.0], [-0.005, 0.1], [-0.005, -0.1]]
        )
        # self.action_space = spaces.Discrete(self.actions.shape[0])

        # 130 km/h ~= 36 m/s
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, 0., -(np.pi / 4)]),
            high=np.array([np.inf, np.inf, np.pi, 36., np.pi / 4])
        )

        self.init_x = self.df['X'][0]
        self.init_y = self.df['Y'][0]
        self.init_theta = self._get_init_theta()
        self.init_v = 2.0
        self.init_psi = 0.0
        self.init_observation = np.array([self.init_x, self.init_y,
                                          self.init_theta, self.init_v,
                                          self.init_psi])
        self.reward = 0.0
        self.last_point = None
        self.state = None
        self.simulator = None
        self.viewer = None

    @staticmethod
    def _read_df(filename):
        df = pd.read_csv(filename)
        df['X'] = df.apply(lambda row: geodetic_to_geocentric(row['LAT'], row['LON'])[0], axis=1)
        df['Y'] = df.apply(lambda row: geodetic_to_geocentric(row['LAT'], row['LON'])[1], axis=1)

        return df[['X', 'Y']]

    def _create_track(self):
        track = Polygon([(row['X'], row['Y']) for index, row in self.df.iterrows()])
        return geopandas.GeoSeries(track)

    def _get_init_theta(self):
        x0 = self.df['X'][0]
        y0 = self.df['Y'][0]
        x1 = self.df['X'][1]
        y1 = self.df['Y'][1]
        theta = math.atan2(y1 - y0, x1 - x0)

        return theta

    def step(self, action):
        # action = [delta_steering_angle, delta_velocity]
        action = self.actions[action]

        done = False
        info = {
            'current_state_x': self.state[0],
            'current_state_y': self.state[1],
            'current_state_theta': self.state[2],
            'current_state_v': self.state[3],
            'current_state_psi': self.state[4],
            'action_delta_angle': action[0],
            'action_delta_v': action[1]
        }

        new_v = self.state[3] + action[1]
        delta_psi = action[0]
        new_x, new_y, new_theta, new_psi = self.simulator.change_state(velocity=new_v, steering_rate=delta_psi)

        new_point = geopandas.GeoSeries([Point(new_x, new_y)])
        distance = self.track.boundary.distance(new_point)[0]

        self.reward -= 0.1
        reward = self.reward

        closest_point = find_closest_point(new_x, new_y, self.df, self.last_point)
        self.last_point = closest_point

        reward += self.last_point

        # based on the distance from the ref trace
        reward -= ((self.tolerance - distance) * 10.0) ** 2

        if distance > self.tolerance:
            reward = -100
            done = True

        # either the vehicle has reached the last point of the ref trace or it
        # has already come past that point, in which case the point_diff would
        # be negative value
        elif self.last_point == self.df.shape[0] - 1:
            done = True

        # update info (include previous state, new state and reward)
        info['closest_point'] = self.last_point
        info['distance'] = distance
        info['reward'] = reward
        info['new_state_x'] = new_x
        info['new_state_y'] = new_y
        info['new_state_theta'] = new_theta
        info['new_state_v'] = new_v
        info['new_state_psi'] = new_psi

        self.state = [new_x, new_y, new_theta, new_v, new_psi]
        observation = self.state

        return observation, reward, done, info

    def reset(self):
        # TODO: consider using [0] as init state or random init state
        self.state = self.init_observation
        self.simulator = Simulator(self.init_x, self.init_y, self.init_theta, self.init_psi)
        self.reward = 0.0
        self.last_point = 0

        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    env = CarEnv()
    for i_episode in range(10):
        env.reset()
        for t in range(10000):
            # env.render()
            action = random.randrange(env.actions.shape[0])
            observation, reward, done, info = env.step(action)
            print(info)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
