#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from gym import Env, spaces
from shapely.geometry import Polygon, Point
from ML.RL.car_env.simulator import Simulator
from ML.RL.car_env.utils import geodetic_to_geocentric
from ML.RL.car_env.utils import find_closest_point
from PIL import Image, ImageDraw
import random


WINDOW_W = 1200
WINDOW_H = 1200
STATE_W = 96
STATE_H = 96
COLOR_GRASS = (65, 152, 10)
COLOR_GRASS_SPOT = (102, 172, 58)
COLOR_TRACK = (102, 102, 102)
BLACK_MARGIN_SIZE = 50
GRASS_MARGIN_SIZE = 50
CAR_FRONT_DIST = 3.     # meters
CAR_BACK_DIST = 1.      # meters
CAR_CORNER_DIST = 1.    # meters


class CarEnv(Env):
    def __init__(self, type='continuous', action_dim=2, actions=None, tolerance=3.0, filename='../../data/ref1.csv'):
        # tolerance in meters
        self.type = type
        self.action_dim = action_dim
        self.tolerance = tolerance

        self.df = self._read_df(filename)
        self.track = self._create_track()

        # steering cmd a.k.a. psi and velocity
        # 0.005 rad ~= 3 deg
        # 0.1 m/s per step
        # self.actions = actions or np.array(
        #     [[0.0, 0.0], [0.0, 0.1], [0.0, -0.1],
        #      [0.005, 0.0], [0.005, 0.1], [0.005, -0.1],
        #      [-0.005, 0.0], [-0.005, 0.1], [-0.005, -0.1]]
        # )
        self.actions = actions or np.array(
            [[0.0, 0.0], [0.0, 0.1], [0.0, -0.1],
             [0.01, 0.0], [0.01, 0.1], [0.01, -0.1],
             [-0.01, 0.0], [-0.01, 0.1], [-0.01, -0.1]]
        )

        # 120 km/h ~= 33 m/s
        self.action_space = spaces.Box(
            np.array([-(np.pi / 6), 0.]), np.array([(np.pi / 6), 33.]), dtype=np.float32
        )  # angle, velocity

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.init_x = self.df['X'][0]
        self.init_y = self.df['Y'][0]
        self.init_theta = self._get_init_theta()
        self.init_v = 2.0
        self.init_psi = 0.0
        # position contains real world values not transformed to WINDOW sizes range
        self.init_position = np.array([self.init_x, self.init_y,
                                       self.init_theta, self.init_v,
                                       self.init_psi])
        self.position = self.init_position
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.x_new_range = WINDOW_W - 2 * (GRASS_MARGIN_SIZE + BLACK_MARGIN_SIZE)
        self.y_new_range = WINDOW_H - 2 * (GRASS_MARGIN_SIZE + BLACK_MARGIN_SIZE)
        self.x_old_range = self.df['X'].max() - self.df['X'].min()
        self.y_old_range = self.df['Y'].max() - self.df['Y'].min()
        self._set_offsets_and_ranges()
        self.full_window = self._get_full_window()
        self.init_state = self._get_state()
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
        return gpd.GeoSeries(track)

    def _get_init_theta(self):
        x0 = self.df['X'][0]
        y0 = self.df['Y'][0]
        x1 = self.df['X'][1]
        y1 = self.df['Y'][1]
        theta = math.atan2(y1 - y0, x1 - x0)

        return theta

    def _set_offsets_and_ranges(self):
        xy_ratio = self.x_old_range / self.y_old_range

        if xy_ratio > 0:
            self.y_new_range = self.y_new_range / xy_ratio
            self.y_offset = (self.x_new_range - self.y_new_range) // 2

        elif xy_ratio < 0:
            self.x_new_range = self.x_new_range / xy_ratio
            self.x_offset = (self.y_new_range - self.x_new_range) // 2

    def _new_x(self, value):
        return ((((value - self.df['X'].min()) * self.x_new_range) / self.x_old_range)
                + GRASS_MARGIN_SIZE + BLACK_MARGIN_SIZE + self.x_offset)

    def _new_y(self, value):
        y = ((((value - self.df['Y'].min()) * self.y_new_range) / self.y_old_range)
             + GRASS_MARGIN_SIZE + BLACK_MARGIN_SIZE + self.y_offset)
        return WINDOW_W - y

    def _get_full_window(self):
        in_polygon = self.track.buffer(-self.tolerance)
        out_polygon = self.track.buffer(self.tolerance)

        in_coords = [list(map(np.float, x.split(' '))) for x in str(in_polygon[0]).strip('POLYGON ()').split(', ')]
        out_coords = [list(map(np.float, x.split(' '))) for x in str(out_polygon[0]).strip('POLYGON ()').split(', ')]

        img = Image.new('RGB', (WINDOW_H, WINDOW_W), 'black')
        draw = ImageDraw.Draw(img)

        draw.polygon([(BLACK_MARGIN_SIZE, BLACK_MARGIN_SIZE),
                      (WINDOW_W - BLACK_MARGIN_SIZE, BLACK_MARGIN_SIZE),
                      (WINDOW_W - BLACK_MARGIN_SIZE, WINDOW_H - BLACK_MARGIN_SIZE),
                      (BLACK_MARGIN_SIZE, WINDOW_H - BLACK_MARGIN_SIZE)],
                     fill=COLOR_GRASS, outline=COLOR_GRASS)

        draw.polygon([(self._new_x(x[0]),
                       self._new_y(x[1]))
                      for x in out_coords], fill=COLOR_TRACK, outline=COLOR_TRACK)
        draw.polygon([(self._new_x(x[0]),
                       self._new_y(x[1]))
                      for x in in_coords], fill=COLOR_GRASS, outline=COLOR_GRASS)

        window = np.array(img)

        return window

    def _get_car_coords(self):
        x = self.position[0]
        y = self.position[1]
        theta = self.position[2]

        front_x = x + CAR_FRONT_DIST * np.cos(theta)
        front_y = y + CAR_FRONT_DIST * np.sin(theta)
        back_x = x - CAR_BACK_DIST * np.cos(theta)
        back_y = y - CAR_BACK_DIST * np.sin(theta)

        left_front_x = self._new_x(front_x + CAR_CORNER_DIST * np.sin(theta))
        left_front_y = self._new_y(front_y - CAR_CORNER_DIST * np.cos(theta))
        right_front_x = self._new_x(front_x - CAR_CORNER_DIST * np.sin(theta))
        right_front_y = self._new_y(front_y + CAR_CORNER_DIST * np.cos(theta))
        left_back_x = self._new_x(back_x - CAR_CORNER_DIST * np.sin(theta))
        left_back_y = self._new_y(back_y + CAR_CORNER_DIST * np.cos(theta))
        right_back_x = self._new_x(back_x + CAR_CORNER_DIST * np.sin(theta))
        right_back_y = self._new_y(back_y - CAR_CORNER_DIST * np.cos(theta))

        return [(left_front_x, left_front_y), (right_front_x, right_front_y),
                (left_back_x, left_back_y), (right_back_x, right_back_y)]

    def step(self, action):
        # action = [delta_steering_angle, delta_velocity]

        done = False
        info = {
            'action_delta_angle': action[0],
            'action_delta_v': action[1]
        }

        new_v = self.position[3] + action[1]
        delta_psi = action[0]
        new_x, new_y, new_theta, new_psi = self.simulator.change_state(velocity=new_v, steering_rate=delta_psi)

        new_point = gpd.GeoSeries([Point(new_x, new_y)])
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
        info['velocity'] = new_v
        info['steering_angle'] = new_psi
        info['closest_point'] = self.last_point
        info['distance'] = distance
        info['reward'] = reward

        self.position = [new_x, new_y, new_theta, new_v, new_psi]
        self.state = self._get_state()
        observation = self.state

        return observation, reward, done, info

    def _get_state(self):
        car_coords = self._get_car_coords()

        img = Image.fromarray(self.full_window)
        draw = ImageDraw.Draw(img)
        draw.polygon(car_coords, fill='red', outline='red')

        car_x = round(self._new_x(self.position[0]))
        car_y = round(self._new_y(self.position[1]))
        min_x = car_x - (STATE_W // 2)
        max_x = car_x + (STATE_W // 2)
        min_y = car_y - (STATE_H // 2)
        max_y = car_y + (STATE_H // 2)

        state = np.array(img)
        state = state[min_y:max_y, min_x:max_x]

        return state

    def reset(self):
        self.position = self.init_position
        self.state = self.init_state
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
            action_i = random.randrange(env.actions.shape[0])
            action = env.actions[action_i]
            observation, reward, done, info = env.step(action)
            print(info)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
