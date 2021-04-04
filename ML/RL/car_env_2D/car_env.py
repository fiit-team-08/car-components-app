#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from gym import Env, spaces
from shapely.geometry import Polygon, Point
from simulator import Simulator
from utils import geodetic_to_geocentric
from utils import find_closest_point
from PIL import Image, ImageDraw
import random
import imageio


WINDOW_W = 2000
WINDOW_H = 2000
STATE_W = 96
STATE_H = 96
COLOR_GRASS = (102, 204, 102)
COLOR_TRACK = (102, 102, 102)
COLOR_WINDOW = (114, 164, 212)
COLOR_CAR = (255, 50, 0)
BLACK_MARGIN_SIZE = 50
GRASS_MARGIN_SIZE = 50
CAR_FRONT_DIST = 3.  # meters
CAR_BACK_DIST = 1.  # meters
CAR_CORNER_DIST = 1.  # meters
TIRE_WIDTH = 0.4  # meters
WINDOW_CORNER_DIST = 0.65 # meters


class CarEnv(Env):
    """
    Environment class containing all the methods necessary for reinforcement
    learning.

    Attributes
    ----------
    df (pd.DataFrame): DataFrame containing info about the ref. trace.
    track (gpd.GeoSeries): Geometric representing the ref. trace.
    actions (list): List of possible discrete actions.
    action_space (np.array): Describes the set of all possible actions by min
        and max values for each element of the action array.
        Action consists of 2 elements: velocity and psi.
    observation_space (np.array): Describes the set of all possible states.
        In this case RGB image with size 96x96.
    init_x (float): Initial x position for reset.
    init_y (float): Initial y position for reset.
    init_theta (float): Initial vehicle heading angle for reset (radians).
    init_v (float): Initial velocity of the vehicle for reset (m/s).
    init_psi (float): Initial steering command of the vehicle given in radians.
    init_position (np.array): Initial position of the car.
    position (np.array): Current position of the car described by 5 elements:
        x: x coord in meters
        y: y coord in meters
        theta: vehicle heading angle in radians
        v: velocity in m/s
        psi: steering angle in radians
    x_offset (float): Offset on the x axis for the drawing of a track.
    y_offset (float): Offset on the y axis for the drawing of a track.
    x_new_range (int): X range for the local state (used for image drawing).
    y_new_range (int): Y range for the local state (used for image drawing).
    x_old_range (float): X range for the global state.
    y_old_range (float): Y range for the global state.
    full_window (np.array): Image of the track in large window.
    init_state (np.array): Initial state used to reset the environment.
    reward (float): Negative reward accumulated with each step taken.
    last_point (int): Index of last passed point on the ref. trace.
    state (np.array or None): Current state of the car.
    simulator (Simulator or None): Instance of Simulator for calculations
        of the new car positions.
    steps_still (int): Count of the steps which didn't alter the position
        (last_point) of the car.
    """

    def __init__(self, type='continuous', action_dim=2, actions=None,
                 tolerance=3.0, filename='../../data/ref1.csv'):
        self.type = type
        self.action_dim = action_dim
        self.tolerance = tolerance  # meters

        self.df = self._read_df(filename)
        self.track = self._create_track()

        # steering cmd a.k.a. psi and velocity
        # 0.005 rad ~= 3 deg
        # 1 m/s per step which is 0.1s (so the speed only changes by 0.1)
        # self.actions = actions or np.array(
        #     [[0.0, 0.0], [0.0, 1.], [0.0, -1.],
        #      [0.005, 0.0], [0.005, 1.], [0.005, -1.],
        #      [-0.005, 0.0], [-0.005, 1.], [-0.005, -1.]]
        # )

        self.actions = actions or np.array(
            [[0.0, 0.0], [0.0, 1.], [0.0, -1.],
             [0.01, 0.0], [0.01, 1.], [0.01, -1.],
             [-0.01, 0.0], [-0.01, 1.], [-0.01, -1.]]
        )

        # 120 km/h ~= 33 m/s
        # angle, velocity
        self.action_space = spaces.Box(
            np.array([-(np.pi / 6), 0.]),
            np.array([(np.pi / 6), 33.]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.init_x = self.df['X'][0]
        self.init_y = self.df['Y'][0]
        self.init_theta = self._get_init_theta()
        self.init_v = 3.0
        self.init_psi = 0.0
        # position contains real world values not transformed to WINDOW range
        self.init_position = np.array([self.init_x, self.init_y,
                                       self.init_theta, self.init_v,
                                       self.init_psi])
        self.position = self.init_position
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.x_new_range = WINDOW_W - 2 * (GRASS_MARGIN_SIZE +
                                           BLACK_MARGIN_SIZE)
        self.y_new_range = WINDOW_H - 2 * (GRASS_MARGIN_SIZE +
                                           BLACK_MARGIN_SIZE)
        self.x_old_range = self.df['X'].max() - self.df['X'].min()
        self.y_old_range = self.df['Y'].max() - self.df['Y'].min()
        self._set_offsets_and_ranges()
        self.full_window = self._get_full_window()
        self.init_state = self._get_state()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.last_point = None
        self.state = None
        self.simulator = None
        self.steps_still = 0
        # start of pseudo rendering
        self.gif_images = list()
        self.episode = 1
        # end of pseudo rendering

    @staticmethod
    def _read_df(filename):
        df = pd.read_csv(filename)
        df['X'] = df.apply(
            lambda row: geodetic_to_geocentric(row['LAT'], row['LON'])[0],
            axis=1
        )
        df['Y'] = df.apply(
            lambda row: geodetic_to_geocentric(row['LAT'], row['LON'])[1],
            axis=1
        )

        return df[['X', 'Y']]

    def _create_track(self):
        track = Polygon(
            [(row['X'], row['Y']) for index, row in self.df.iterrows()]
        )

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

    # transform x from global (real world) coordinates to local
    def _new_x(self, value):
        return ((((value - self.df['X'].min()) * self.x_new_range)
                 / self.x_old_range)
                + GRASS_MARGIN_SIZE + BLACK_MARGIN_SIZE + self.x_offset)

    # transform y from global (real world) coordinates to local
    def _new_y(self, value):
        y = ((((value - self.df['Y'].min()) * self.y_new_range)
              / self.y_old_range)
             + GRASS_MARGIN_SIZE + BLACK_MARGIN_SIZE + self.y_offset)
        return WINDOW_W - y

    # draw large image of the track
    def _get_full_window(self):
        in_polygon = self.track.buffer(-self.tolerance)
        out_polygon = self.track.buffer(self.tolerance)

        in_coords = [list(map(np.float, x.split(' '))) for x in
                     str(in_polygon[0]).strip('POLYGON ()').split(', ')]
        out_coords = [list(map(np.float, x.split(' '))) for x in
                      str(out_polygon[0]).strip('POLYGON ()').split(', ')]

        img = Image.new('RGB', (WINDOW_H, WINDOW_W), 'black')
        draw = ImageDraw.Draw(img)

        # draw green field
        draw.polygon(
            [(BLACK_MARGIN_SIZE, BLACK_MARGIN_SIZE),
             (WINDOW_W - BLACK_MARGIN_SIZE, BLACK_MARGIN_SIZE),
             (WINDOW_W - BLACK_MARGIN_SIZE, WINDOW_H - BLACK_MARGIN_SIZE),
             (BLACK_MARGIN_SIZE, WINDOW_H - BLACK_MARGIN_SIZE)],
            fill=COLOR_GRASS, outline=COLOR_GRASS)

        # draw track
        draw.polygon([(self._new_x(x[0]),
                       self._new_y(x[1]))
                      for x in out_coords], fill=COLOR_TRACK,
                     outline=COLOR_TRACK)
        draw.polygon([(self._new_x(x[0]),
                       self._new_y(x[1]))
                      for x in in_coords], fill=COLOR_GRASS,
                     outline=COLOR_GRASS)

        # convert to np.array
        window = np.array(img)

        return window

    # car coordinates for drawing of the state
    def _get_car_coords(self, original=False):
        x = self.position[0]
        y = self.position[1]
        theta = self.position[2]

        front_x = x + CAR_FRONT_DIST * np.cos(theta)
        front_y = y + CAR_FRONT_DIST * np.sin(theta)
        back_x = x - CAR_BACK_DIST * np.cos(theta)
        back_y = y - CAR_BACK_DIST * np.sin(theta)

        right_front_x = front_x + CAR_CORNER_DIST * np.sin(theta)
        right_front_y = front_y - CAR_CORNER_DIST * np.cos(theta)
        left_front_x = front_x - CAR_CORNER_DIST * np.sin(theta)
        left_front_y = front_y + CAR_CORNER_DIST * np.cos(theta)
        left_back_x = back_x - CAR_CORNER_DIST * np.sin(theta)
        left_back_y = back_y + CAR_CORNER_DIST * np.cos(theta)
        right_back_x = back_x + CAR_CORNER_DIST * np.sin(theta)
        right_back_y = back_y - CAR_CORNER_DIST * np.cos(theta)

        if original:
            return [
                (left_front_x, left_front_y), (right_front_x, right_front_y),
                (right_back_x, right_back_y), (left_back_x, left_back_y)
            ]
        else:
            return [
                (self._new_x(left_front_x), self._new_y(left_front_y)),
                (self._new_x(right_front_x), self._new_y(right_front_y)),
                (self._new_x(right_back_x), self._new_y(right_back_y)),
                (self._new_x(left_back_x), self._new_y(left_back_y))
            ]

    def _get_tire_coords(self):
        theta = self.position[2]
        delta = theta + self.position[4]
        car_coords = self._get_car_coords(original=True)
        car = {
            'left_front': car_coords[0],
            'right_front': car_coords[1],
            'right_back': car_coords[2],
            'left_back': car_coords[3]
        }

        tires = list()
        pos_and_angle = list()

        # left front tire
        x = (car['left_back'][0]
             + (car['left_front'][0] - car['left_back'][0]) * 0.75)
        y = (car['left_back'][1]
             + (car['left_front'][1] - car['left_back'][1]) * 0.75)
        pos_and_angle.append({'x': x, 'y': y, 'angle': delta})

        # right front tire
        x = (car['right_back'][0]
             + (car['right_front'][0] - car['right_back'][0]) * 0.75)
        y = (car['right_back'][1]
             + (car['right_front'][1] - car['right_back'][1]) * 0.75)
        pos_and_angle.append({'x': x, 'y': y, 'angle': delta})

        # right back tire
        x = (car['right_back'][0]
             + (car['right_front'][0] - car['right_back'][0]) * 0.25)
        y = (car['right_back'][1]
             + (car['right_front'][1] - car['right_back'][1]) * 0.25)
        pos_and_angle.append({'x': x, 'y': y, 'angle': theta})

        # left back tire
        x = (car['left_back'][0]
             + (car['left_front'][0] - car['left_back'][0]) * 0.25)
        y = (car['left_back'][1]
             + (car['left_front'][1] - car['left_back'][1]) * 0.25)
        pos_and_angle.append({'x': x, 'y': y, 'angle': theta})

        for tire in pos_and_angle:
            front_x = tire['x'] + 0.5 * np.cos(tire['angle'])
            front_y = tire['y'] + 0.5 * np.sin(tire['angle'])
            back_x = tire['x'] - 0.5 * np.cos(tire['angle'])
            back_y = tire['y'] - 0.5 * np.sin(tire['angle'])

            right_front_x = self._new_x(
                front_x + (TIRE_WIDTH / 2) * np.sin(tire['angle']))
            right_front_y = self._new_y(
                front_y - (TIRE_WIDTH / 2) * np.cos(tire['angle']))
            left_front_x = self._new_x(
                front_x - (TIRE_WIDTH / 2) * np.sin(tire['angle']))
            left_front_y = self._new_y(
                front_y + (TIRE_WIDTH / 2) * np.cos(tire['angle']))
            left_back_x = self._new_x(
                back_x - (TIRE_WIDTH / 2) * np.sin(tire['angle']))
            left_back_y = self._new_y(
                back_y + (TIRE_WIDTH / 2) * np.cos(tire['angle']))
            right_back_x = self._new_x(
                back_x + (TIRE_WIDTH / 2) * np.sin(tire['angle']))
            right_back_y = self._new_y(
                back_y - (TIRE_WIDTH / 2) * np.cos(tire['angle']))

            tires.append(
                [(left_front_x, left_front_y), (right_front_x, right_front_y),
                 (right_back_x, right_back_y), (left_back_x, left_back_y)]
            )

        return tires

    def _get_car_windows_coords(self):
        x = self.position[0]
        y = self.position[1]
        theta = self.position[2]

        windows = list()
        positions = list()

        front_x = x + 1.8 * np.cos(theta)
        front_y = y + 1.8 * np.sin(theta)
        back_x = x + 1. * np.cos(theta)
        back_y = y + 1. * np.sin(theta)
        positions.append([front_x, front_y, back_x, back_y])

        front_x = x
        front_y = y
        back_x = x - 0.3 * np.cos(theta)
        back_y = y - 0.3 * np.sin(theta)
        positions.append([front_x, front_y, back_x, back_y])

        for pos in positions:
            right_front_x = self._new_x(
                pos[0] + WINDOW_CORNER_DIST * np.sin(theta))
            right_front_y = self._new_y(
                pos[1] - WINDOW_CORNER_DIST * np.cos(theta))
            left_front_x = self._new_x(
                pos[0] - WINDOW_CORNER_DIST * np.sin(theta))
            left_front_y = self._new_y(
                pos[1] + WINDOW_CORNER_DIST * np.cos(theta))
            left_back_x = self._new_x(
                pos[2] - WINDOW_CORNER_DIST * np.sin(theta))
            left_back_y = self._new_y(
                pos[3] + WINDOW_CORNER_DIST * np.cos(theta))
            right_back_x = self._new_x(
                pos[2] + WINDOW_CORNER_DIST * np.sin(theta))
            right_back_y = self._new_y(
                pos[3] - WINDOW_CORNER_DIST * np.cos(theta))

            windows.append([
                (left_front_x, left_front_y), (right_front_x, right_front_y),
                (right_back_x, right_back_y), (left_back_x, left_back_y)
            ])

        return windows

    def step(self, action):
        # action = [delta_steering_angle, delta_velocity]
        """
        Applies the action given as parameter and calculates the new state
        (observation).

        Parameters
        ----------
        action (np.array) : Action that should be performed.
            Consists of [delta_psi, delta_v], where delta_psi is change of
            the steering command in radians and delta_v is change of
            velocity in m/s.

        Returns
        -------
        observation (np.array): New state of the car.
        reward (float): Reward for the performed action.
        done (bool): True when the car moves further than 0.5m away from
            the ref. trace, otherwise false.
        info (dict): Information about new state, action and reward.
        """

        done = False
        info = {
            'action_delta_angle': action[0],
            'action_delta_v': action[1]
        }

        new_v = self.position[3] + action[1]
        delta_psi = -action[0]
        new_x, new_y, new_theta, new_psi = self.simulator.change_state(
            velocity=new_v, steering_rate=delta_psi
        )

        new_point = gpd.GeoSeries([Point(new_x, new_y)])
        distance = self.track.boundary.distance(new_point)[0]
        closest_point = find_closest_point(new_x, new_y, self.df,
                                           self.last_point)

        self.reward -= 0.1
        reward = self.reward - self.prev_reward
        reward += closest_point

        if closest_point == self.last_point:
            self.steps_still += 1

        # based on the distance from the ref. trace
        if self.tolerance < 1.:
            reward -= distance
            # reward -= distance * 10
            # reward += ((self.tolerance - distance) * 10.0) ** 2
        else:
            reward -= distance
            # reward += (self.tolerance - distance) ** 2

        if distance > self.tolerance:
            reward = -100
            done = True

        elif self.steps_still > 10:
            reward = -100
            done = True

        elif new_v < 0.0:
            reward = -100
            done = True

        elif not (-np.pi/6 <= new_psi <= np.pi/6):
            reward = -100
            done = True

        # check if the car reached the last point of the track
        elif closest_point == self.df.shape[0] - 1:
            done = True

        self.last_point = closest_point
        self.prev_reward = reward

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

    # get image of the state according to the new position of the car
    def _get_state(self):
        car_coords = self._get_car_coords()
        tires = self._get_tire_coords()
        car_windows = self._get_car_windows_coords()

        # load large window
        img = Image.fromarray(self.full_window)
        draw = ImageDraw.Draw(img)
        # draw car
        draw.polygon(car_coords, fill=COLOR_CAR, outline=COLOR_CAR)
        for window in car_windows:
            draw.polygon(window, fill=COLOR_WINDOW, outline=COLOR_WINDOW)
        for tire in tires:
            draw.polygon(tire, fill='black', outline='black')

        car_x = round(self._new_x(self.position[0]))
        car_y = round(self._new_y(self.position[1]))
        min_x = car_x - (STATE_W // 2)
        max_x = car_x + (STATE_W // 2)
        min_y = car_y - (STATE_H // 2)
        max_y = car_y + (STATE_H // 2)

        # crop the large window to 96x96 pixels
        state = np.array(img)
        state = state[min_y:max_y, min_x:max_x]

        return state

    def reset(self):
        """
        Resets the state of the car to the initial state.
        """

        self.position = self.init_position
        self.state = self.init_state
        self.simulator = Simulator(self.init_x, self.init_y,
                                   self.init_theta, self.init_psi)
        self.reward = 0.0
        self.prev_reward = 0.0
        self.last_point = 0
        self.steps_still = 0

        # start of pseudo rendering
        if self.gif_images:
            imageio.mimsave('episode{}.gif'.format(self.episode),
                            self.gif_images, fps=8)
            self.episode += 1
            self.gif_images = list()
        # end of pseudo rendering

        return self.state

    def render(self, mode='human'):
        # start of pseudo rendering
        file_path = 'image.png'

        img = Image.fromarray(self.state)
        img.save(file_path, 'png')
        self.gif_images.append(imageio.imread(file_path))
        # end of pseudo rendering

    def close(self):
        # start of pseudo rendering
        if self.gif_images:
            imageio.mimsave('episode{}.gif'.format(self.episode),
                            self.gif_images, fps=16)
        # end of pseudo rendering


if __name__ == '__main__':
    env = CarEnv()
    for i_episode in range(10):
        env.reset()
        for t in range(1000):
            env.render()
            action_i = random.randrange(env.actions.shape[0])
            action = env.actions[action_i]
            observation, reward, done, info = env.step(action)
            print(info)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
