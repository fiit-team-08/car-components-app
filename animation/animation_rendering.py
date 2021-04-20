# https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
import math
import copy
import json

import gym
from gym import spaces
import pandas as pd
import numpy as np
import shapely.geometry as geom
from obspy.geodetics import degrees2kilometers
#from analysis.obspy_copy import degrees2kilometers

from gym.envs.classic_control import rendering
from pyglet.window import key, mouse

import pyglet
from pyglet import gl

WINDOW_W = 800
WINDOW_H = 800

SCALE = 1  # Track scale in the viewing window

CAR_WIDTH = 1.680 * SCALE
CAR_LENGTH = 2.817 * SCALE
WHEEL_SPACING = 1.480 * SCALE
WHEEL_LENGTH = 0.3 * SCALE
WHEEL_WIDTH = 0.195 * SCALE
WHEEL_BASE = 2.345 * SCALE

CAR_COLOR = (1, 0, 0)  # RED
WINDOW_COLOR = (.5, .5, .8)  # BLUE
WHEEL_COLOR = (0, 0, 0)  # BLACK

ROAD_WIDTH = 7

ZOOM = 50

env = None
close = False

# environment with continuous observation space
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

    debug = '../../ML/data/ref1.csv'
    production = 'benchmark/ref1.csv'

    def __init__(self, filename=production, type='continuous',
                 action_dim=2, verbose=1):

        super(CarEnv, self).__init__()
        self.df = self._read_df(filename)

        self.road = self._create_road()
        self.map_center = self._get_track_start()

        self._create_road_borders()

        self.init_x = 0
        self.init_y = 0
        self.init_theta = 0
        self.init_v = 2.0
        self.init_psi = 0.0
        self.time = 0
        self.init_observation = np.array([
            self.init_x, 
            self.init_y,
            self.init_v,
            self.init_theta, 
            self.init_psi,
            self.time
        ])
        self.state = self.init_observation
        self.const_t = False
        self.const_v = False

        self.track = None
        self.outer_border = None
        self.inner_border = None

        self.car_trans = None
        self.left_wheel_trans = None
        self.right_wheel_trans = None
        self.left_rearwheel_trans = None
        self.right_rearwheel_trans = None
        self.track_trans = None

        self.viewer = None

        self.zoom = 0
        self.is_zoomed = False
        self.car_rendered = False

    @staticmethod
    def _read_df(filename: str) -> pd.DataFrame:
        df = pd.read_csv(filename)
        # covert degrees to meters and degrees to radians
        df['LAT'] = df['LAT'].apply(lambda deg: degrees2kilometers(deg) * 1000)
        df['LON'] = df['LON'].apply(lambda deg: degrees2kilometers(deg) * 1000)
        df['CRS'] = df['CRS'].apply(lambda deg: np.deg2rad(deg))

        return df

    def _create_road(self):
        """
        Creates road from given points
        """
        self.points = list()

        for index, row in self.df.iterrows():
            self.points.append(geom.Point(row['LON'], row['LAT']))

        return geom.LineString(self.points)

    def _create_road_borders(self):
        """
        Creates point forming borders (inner anf outer) 
        of track with given road width
        """
        self.outer = list()
        self.inner = list()

        for i, point in enumerate(self.points):

            x_center, y_center = self.map_center
            dist = ROAD_WIDTH/2  # half_width of road

            AL = []  # transformed point A to the left
            AR = []  # transformed point A to the right
            BL = []  # transformed point B to the left
            BR = []  # transformed point B to the right

            A = [point.x, point.y]  # start point
            if i == len(self.points) - 1:
                B = [self.points[0].x, self.points[0].y]  # end point
            else:
                B = [self.points[i + 1].x, self.points[i + 1].y]  # end point

            V = [B[0] - A[0], B[1] - A[1]]  # vector of points A and B

            # angle between vector V and X axis in coordinate system
            cos_alfa = math.acos(V[0] / math.sqrt(V[0]**2 + V[1]**2))

            # if angle is greater than 90 degrees, subtract 90 degerees
            if cos_alfa > (math.pi/2):
                beta = cos_alfa - (math.pi/2)
            else:
                beta = (math.pi/2) - cos_alfa

            delta_x = dist * math.cos(beta)  # difference on X axis
            delta_y = dist * math.sin(beta)  # difference on Y axis

            # all possible directions of vector V
            if V[0] > 0:  # direction to right
                if V[1] == 0:
                    AL = [A[0], A[1] + dist]
                    AR = [A[0], A[1] - dist]
                    BL = [B[0], B[1] + dist]
                    BR = [B[0], B[1] - dist]
                if V[1] > 0:  # direction right, up
                    AL = [A[0] - delta_x, A[1] + delta_y]
                    AR = [A[0] + delta_x, A[1] - delta_y]
                    BL = [B[0] - delta_x, B[1] + delta_y]
                    BR = [B[0] + delta_x, B[1] - delta_y]
                if V[1] < 0:  # direction right, down
                    AL = [A[0] + delta_x, A[1] + delta_y]
                    AR = [A[0] - delta_x, A[1] - delta_y]
                    BL = [B[0] + delta_x, B[1] + delta_y]
                    BR = [B[0] - delta_x, B[1] - delta_y]

            if V[0] < 0:  # direction to left
                if V[1] == 0:
                    AL = [A[0], A[1] - dist]
                    AR = [A[0], A[1] + dist]
                    BL = [B[0], B[1] - dist]
                    BR = [B[0], B[1] + dist]
                if V[1] > 0:  # direction left, up
                    AL = [A[0] - delta_x, A[1] - delta_y]
                    AR = [A[0] + delta_x, A[1] + delta_y]
                    BL = [B[0] - delta_x, B[1] - delta_y]
                    BR = [B[0] + delta_x, B[1] + delta_y]
                if V[1] < 0:  # direction left, down
                    AL = [A[0] + delta_x, A[1] - delta_y]
                    AR = [A[0] - delta_x, A[1] + delta_y]
                    BL = [B[0] + delta_x, B[1] - delta_y]
                    BR = [B[0] - delta_x, B[1] + delta_y]

            if V[1] > 0 and V[0] == 0:  # direction up
                AL = [A[0] - dist, A[1]]
                AR = [A[0] + dist, A[1]]
                BL = [B[0] - dist, B[1]]
                BR = [B[0] + dist, B[1]]
            if V[1] < 0 and V[0] == 0:  # direction down
                AL = [A[0] + dist, A[1]]
                AR = [A[0] - dist, A[1]]
                BL = [B[0] + dist, B[1]]
                BR = [B[0] - dist, B[1]]

            self.outer.append(((AL[0] - x_center) * SCALE, (AL[1] - y_center) * SCALE))
            self.outer.append(((BL[0] - x_center) * SCALE, (BL[1] - y_center) * SCALE))
            self.inner.append(((AR[0] - x_center) * SCALE, (AR[1] - y_center) * SCALE))
            self.inner.append(((BR[0] - x_center) * SCALE, (BR[1] - y_center) * SCALE))

    def _get_track_start(self):
        """
        Gets track centre based on starting point of the track
        """
        x = self.points[0].x
        y = self.points[0].y
        
        return ((x, y))

    def _create_track(self):
        """
        Transforms world coordinates into x,y coordinates by finding center
        and translating points and creates track represented by x,y points

        :returns - list of x,y points representing track in
        """
        center_x, center_y = self.map_center

        # append all x,y points to the track list
        coordinates = []
        for point in self.points:
            x = point.x - center_x
            y = point.y - center_y

            coordinates.append((x * SCALE, y * SCALE))

        coordinates.append(((self.points[0].x - center_x) * SCALE, (self.points[0].y - center_y) * SCALE))

        return coordinates

    def _create_track_border(self, points):

        coordinates = []
        for point in points:
            x = point[0]
            y = point[1]
            coordinates.append((x, y))

        coordinates.append((points[0][0], points[0][1]))

        return coordinates

    def _create_track_object(self):
        """
        Creats track polygon from precalculated border coordinates
        with given color
        """


        """
        Connecting all points as individual polygons is too slow,
        so connecions are between every 5th point
        """
        for index in range(0,len(self.outer),5):
            if index < (len(self.outer) - 5):
                ll_x, ll_y = self.inner[index]
                ul_x, ul_y = self.inner[index + 5]
                lr_x, lr_y = self.outer[index]
                ur_x, ur_y = self.outer[index + 5]
            else:
                ll_x, ll_y = self.inner[index]
                ul_x, ul_y = self.inner[0]
                lr_x, lr_y = self.outer[index]
                ur_x, ur_y = self.outer[0]


            track = rendering.FilledPolygon([
                    (ll_x + WINDOW_W / 2, ll_y + WINDOW_H / 2),
                    (ul_x + WINDOW_W / 2, ul_y + WINDOW_H / 2),
                    (ur_x + WINDOW_W / 2, ur_y + WINDOW_H / 2),
                    (lr_x + WINDOW_W / 2, lr_y + WINDOW_H / 2)
            ])
            # creates track represented by line
            self.track_trans = rendering.Transform()
            track.add_attr(self.track_trans)
            track.add_attr(self.transform)
            track.set_color(0.25, 0.25, 0.25)
            self.viewer.add_geom(track)




    def _create_car_object(self, car_width, car_length, car_color):
        """
        Creates car object (polygon) which includes body and rear wheels
        """
        car_body = list()

        R, G, B = car_color

        left, right, top, bottom = (
            -car_width / 2,
            car_width / 2,
            car_length / 2,
            -car_length / 2
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
        car.add_attr(self.transform)
        car.set_color(R, G, B)
        car_body.append(car)

        wheel_oversize = WHEEL_WIDTH - (CAR_WIDTH - WHEEL_SPACING)/2

        # REAR WHEELS
        left_rear_wheel = rendering.FilledPolygon([
            (left - wheel_oversize, bottom),
            (left - wheel_oversize, bottom + WHEEL_LENGTH),
            (left + wheel_oversize, bottom + WHEEL_LENGTH),
            (left + wheel_oversize, bottom)
        ])
        left_rear_wheel.add_attr(self.car_trans)
        left_rear_wheel.add_attr(self.transform)
        left_rear_wheel.set_color(.1, .1, .1)
        car_body.append(left_rear_wheel)

        right_rear_wheel = rendering.FilledPolygon([
            (right - wheel_oversize, bottom),
            (right - wheel_oversize, bottom + WHEEL_LENGTH),
            (right + wheel_oversize, bottom + WHEEL_LENGTH),
            (right + wheel_oversize, bottom)
        ])
        right_rear_wheel.add_attr(self.car_trans)
        right_rear_wheel.add_attr(self.transform)
        right_rear_wheel.set_color(.1, .1, .1)
        car_body.append(right_rear_wheel)

        return car_body

    def _define_label(self, text, offset):
        label = pyglet.text.Label(
            text,
            font_size=12,
            x=5,
            y=WINDOW_H - offset,
            color=(255, 255, 255, 255),
        )
        return label

    def _car_indicator_object(self):
        self.car_small = pyglet.shapes.Rectangle(50,  50, 20, 80, color=(1, 0, 0))


    def _create_track_line(self, line, color):
        """
        Function to create middle line and borders of the track
        """
        R,G,B = color
        track_line = rendering.PolyLine(line, 1)
        track_line.add_attr(self.track_trans)
        track_line.add_attr(self.transform)
        track_line.set_color(R, G, B)
        return track_line

    def step(self, action):
        """
        Updates parameters with each step
        """

        new_x = action['x'] * SCALE
        new_y = action['y'] * SCALE

        if 'velocity' in action:
            if action['velocity'] == 'const':
                new_v = 1
                self.const_v = True
            else:
                new_v = action['velocity']
        else:
            new_v = 1

        if 'time' in action:
            new_t = action['time']
        else:
            self.const_t = True
            new_t = self.time


        new_theta = action['heading_angle']
        new_psi = action['steering_angle']
        #new_t = action['time']

        info = {
            'current_state': {
                'x': self.state[0],
                'y': self.state[1],
                'v': self.state[2],
                'theta': self.state[3],
                'psi': self.state[4],
                'time': self.state[5]
            },
        }

        self.state = [new_x, new_y, new_v, new_theta, new_psi, new_t]

        return self.state

    def reset(self):
        """
        Resets the state of the car to the initial state.
        """
        self.state = self.init_observation

        self.track = self._create_track()
        self.outer_border = self._create_track_border(self.outer)
        self.inner_border = self._create_track_border(self.inner)

        return self.init_observation



    def render(self, mode='human'):

        x, y, v, theta, psi, time = self.state
        center_x, center_y = self.map_center

        self.time = self.time + 0.1 if self.const_t else time

        car_w = CAR_WIDTH * ZOOM
        car_l = CAR_LENGTH * ZOOM
        wheel_w = WHEEL_WIDTH * ZOOM
        wheel_l = WHEEL_LENGTH * ZOOM

        if self.viewer is None:
            x = 0
            y = 0

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.transform = rendering.Transform()

            # -----DEFINE LABELS-----
            self.time_label = self._define_label('time', 20)
            self.velocity_label = self._define_label('time', 40)
            self.steering_angle_label = self._define_label('time', 60)
            self.upper_left_indicator = pyglet.shapes.Rectangle(0, WINDOW_H - 80, 200, 80, color=(0,0,0))
            #self.lower_left_indicator = pyglet.shapes.Rectangle(0, 0, 200, 200, color=(200, 200, 200))


            # -----DEFINE STEERING WHEEL MINIATURE-----
            # self.lower_right_indicator = pyglet.shapes.Rectangle(WINDOW_W - 150, 0, 150, 150, color=(200, 200, 200))
            #self.steering_wheel_outer = pyglet.shapes.Circle(WINDOW_W-75, 75, 70, color=(0, 0, 0))
            #self.steering_wheel_inner = pyglet.shapes.Circle(WINDOW_W-75, 75, 55, color=(200, 200, 200))

            # -----DEFINE CAR MINIATURE-----
            self.car_small = pyglet.shapes.Rectangle(100, 100, car_w, car_l, color=(255, 0, 0))
            self.car_small_wheel_left_rear = pyglet.shapes.Rectangle(100, 100, wheel_w, -wheel_l, color=(0, 0, 0))
            self.car_small_wheel_right_rear = pyglet.shapes.Rectangle(100, 100, -wheel_w, -wheel_l, color=(0, 0, 0))
            self.car_small_wheel_left_front = pyglet.shapes.Rectangle(100, 100, wheel_w, wheel_l, color=(0, 0, 0))
            self.car_small_wheel_right_front = pyglet.shapes.Rectangle(100, 100, -wheel_w, wheel_l, color=(0, 0, 0))



            # -----BACKGROUND GRASS-----
            background = rendering.FilledPolygon([
                (-WINDOW_W, -WINDOW_H),
                (-WINDOW_W, WINDOW_H * 2),
                (WINDOW_W * 2, WINDOW_H * 2),
                (WINDOW_W * 2, -WINDOW_H)
            ])
            self.background_trans = rendering.Transform()
            background.add_attr(self.background_trans)
            background.add_attr(self.transform)
            background.set_color(0.051, 0.455, 0.024)
            self.viewer.add_geom(background)



            # -----TRACK AND TRACK LINES RENDERING-----
            self._create_track_object()

            # centers points of the track to the center of the screen
            for i in range(len(self.track)):
                x, y = self.track[i]
                self.track[i] = x + WINDOW_W / 2, y + WINDOW_H / 2

            for i in range(len(self.inner_border)):
                x, y = self.inner_border[i]
                self.inner_border[i] = x + WINDOW_W / 2, y + WINDOW_H / 2

            for i in range(len(self.outer_border)):
                x, y = self.outer_border[i]
                self.outer_border[i] = x + WINDOW_W / 2, y + WINDOW_H / 2

            # creates track represented by line
            self.track_trans = rendering.Transform()
            track_line = self._create_track_line(self.track, color=(1, 1, 1))
            self.viewer.add_geom(track_line)
            track_inner = self._create_track_line(self.inner_border, color=(1, 0, 0))
            self.viewer.add_geom(track_inner)
            track_outer = self._create_track_line(self.outer_border, color=(1, 0, 0))
            self.viewer.add_geom(track_outer)


            # -----CAR OBJECT RENDERING-----
            if self.car_rendered == False:
                car_body = self._create_car_object(CAR_WIDTH, CAR_LENGTH, CAR_COLOR)

                for part in car_body:
                    self.viewer.add_geom(part)

                # FRONT WHEELS

                wheel = rendering.FilledPolygon([
                    (-WHEEL_WIDTH / 2, -WHEEL_LENGTH / 2),
                    (WHEEL_WIDTH / 2, -WHEEL_LENGTH / 2),
                    (WHEEL_WIDTH / 2, WHEEL_LENGTH / 2),
                    (-WHEEL_WIDTH / 2, WHEEL_LENGTH / 2)
                ])

                left_front_wheel = copy.deepcopy(wheel)
                self.left_wheel_trans = rendering.Transform()
                left_front_wheel.add_attr(self.left_wheel_trans)
                left_front_wheel.add_attr(self.car_trans)
                left_front_wheel.add_attr(self.transform)
                left_front_wheel.set_color(.1, .1, .1)
                self.viewer.add_geom(left_front_wheel)

                right_front_wheel = copy.deepcopy(wheel)
                self.right_wheel_trans = rendering.Transform()
                right_front_wheel.add_attr(self.right_wheel_trans)
                right_front_wheel.add_attr(self.car_trans)
                right_front_wheel.add_attr(self.transform)
                right_front_wheel.set_color(.1, .1, .1)
                self.viewer.add_geom(right_front_wheel)

                self.car_rendered = True

                # TODO - Make funcion from this
                # -----MINIATURE OF CAR IN LOWER LEFT INDICATOR-----
                lower_left_indicator = rendering.FilledPolygon([
                    (0, 0),
                    (0, 200),
                    (200, 200),
                    (200, 0)
                ])
                lower_left_indicator.set_color(0.784, 0.784, 0.784)
                self.viewer.add_geom(lower_left_indicator)

                left, right, top, bottom = (
                    -CAR_WIDTH * ZOOM / 2,
                    CAR_WIDTH * ZOOM / 2,
                    CAR_LENGTH * ZOOM / 2,
                    -CAR_LENGTH * ZOOM / 2
                )
                # CAR BODY

                car = rendering.FilledPolygon([
                    (left, bottom),
                    (left, top),
                    (right, top),
                    (right, bottom)
                ])
                self.mini_car_trans = rendering.Transform()
                self.mini_car_trans.set_translation(100,100)
                car.add_attr(self.mini_car_trans)
                car.set_color(1, 0, 0)
                self.viewer.add_geom(car)


                # WHEELS REAR
                wheel_oversize = WHEEL_WIDTH * ZOOM - (CAR_WIDTH * ZOOM - WHEEL_SPACING * ZOOM) / 2

                left_rear_wheel = rendering.FilledPolygon([
                    (left - wheel_oversize, bottom),
                    (left - wheel_oversize, bottom + WHEEL_LENGTH * ZOOM),
                    (left + wheel_oversize, bottom + WHEEL_LENGTH * ZOOM),
                    (left + wheel_oversize, bottom)
                ])

                left_rear_wheel.add_attr(self.mini_car_trans)
                left_rear_wheel.set_color(.1, .1, .1)
                self.viewer.add_geom(left_rear_wheel)

                right_rear_wheel = rendering.FilledPolygon([
                    (right - wheel_oversize, bottom),
                    (right - wheel_oversize, bottom + WHEEL_LENGTH * ZOOM),
                    (right + wheel_oversize, bottom + WHEEL_LENGTH * ZOOM),
                    (right + wheel_oversize, bottom)
                ])
                right_rear_wheel.add_attr(self.mini_car_trans)
                right_rear_wheel.set_color(.1, .1, .1)
                self.viewer.add_geom(right_rear_wheel)

                # WHEELS FRONT MINI
                wheel = rendering.FilledPolygon([
                    (-WHEEL_WIDTH*ZOOM / 2, -WHEEL_LENGTH*ZOOM / 2),
                    (WHEEL_WIDTH*ZOOM / 2, -WHEEL_LENGTH*ZOOM / 2),
                    (WHEEL_WIDTH*ZOOM / 2, WHEEL_LENGTH*ZOOM / 2),
                    (-WHEEL_WIDTH*ZOOM / 2, WHEEL_LENGTH*ZOOM / 2)
                ])

                left_front_wheel_mini = copy.deepcopy(wheel)
                self.mini_left_wheel_trans = rendering.Transform()
                left_front_wheel_mini.add_attr(self.mini_left_wheel_trans)
                left_front_wheel_mini.add_attr(self.mini_car_trans)
                left_front_wheel_mini.set_color(.1, .1, .1)
                self.viewer.add_geom(left_front_wheel_mini)

                right_front_wheel_mini = copy.deepcopy(wheel)
                self.mini_right_wheel_trans = rendering.Transform()
                right_front_wheel_mini.add_attr(self.mini_right_wheel_trans)
                right_front_wheel_mini.add_attr(self.mini_car_trans)
                right_front_wheel_mini.set_color(.1, .1, .1)
                self.viewer.add_geom(right_front_wheel_mini)

        # -----ZOOMING INTO CAR VIEW-----
        if self.zoom == ZOOM:
            self.is_zoomed = True
            # Zooms into car view until reached specific zoom
        if self.is_zoomed is not True:
            if self.time == 0:
                self.time = 0.05
            self.zoom = 0.01 * SCALE * max(1 - self.time, 0) + (ZOOM * SCALE * min(self.time, 1))

        # -----FOLLOWING CAR VIEW-----
        scroll_x = x
        scroll_y = y
        angle = -theta
        self.transform.set_scale(self.zoom, self.zoom)
        self.transform.set_translation(
            -WINDOW_W * (self.zoom / 2 - 0.5) - (scroll_x * self.zoom),
            -WINDOW_H * (self.zoom / 2 - 0.5) - (scroll_y * self.zoom)
        )

        # -----TRANSFORMATION AND ROTATION OF CAR ON TRACK-----

        # car rotation and translation
        self.car_trans.set_translation(x + WINDOW_W / 2, y + WINDOW_H / 2)
        self.car_trans.set_rotation(theta - (math.pi) / 2)

        # wheels rotation and translation
        wheel_oversize = WHEEL_WIDTH - (CAR_WIDTH - WHEEL_SPACING) / 2
        self.left_wheel_trans.set_translation(-WHEEL_SPACING / 2 - wheel_oversize, WHEEL_BASE / 2)
        self.right_wheel_trans.set_translation(WHEEL_SPACING / 2 + wheel_oversize, WHEEL_BASE / 2)
        self.left_wheel_trans.set_rotation(psi)
        self.right_wheel_trans.set_rotation(psi)

        # -----TRANSFORMATION AND ROTATION OF CAR IN LOWER LEFT INDICATOR-----
        wheel_oversize = WHEEL_WIDTH * ZOOM - (CAR_WIDTH * ZOOM - WHEEL_SPACING * ZOOM) / 2
        self.mini_car_trans.set_rotation(theta - (math.pi) / 2)
        self.mini_left_wheel_trans.set_translation(-WHEEL_SPACING * ZOOM / 2 - wheel_oversize, WHEEL_BASE * ZOOM / 2)
        self.mini_right_wheel_trans.set_translation(WHEEL_SPACING * ZOOM / 2 + wheel_oversize, WHEEL_BASE * ZOOM / 2)
        self.mini_left_wheel_trans.set_rotation(psi)
        self.mini_right_wheel_trans.set_rotation(psi)

        # -----RENDERING-----
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()
        win.clear()

        for geom in self.viewer.geoms:
            geom.render()

        # ----------LABELS-----------
        self.upper_left_indicator.draw()
        #self.lower_right_indicator.draw()
        self.time_label.text = f'time: {round(time, 1)}'
        self.time_label.draw()

        self.velocity_label.text = 'velocity: constant' if self.const_v else f'velocity: {v:.2f} km/h'
        self.velocity_label.draw()

        self.steering_angle_label.text = f'steering_angle: {math.degrees(psi):.2f} deg'
        self.steering_angle_label.draw()

        win.flip()
        self.viewer.isopen





    def zoom_in(self):
        self.zoom /= 0.8
        return
    
    def zoom_out(self):
        self.zoom *= 0.8
        return

    def close(self):
        """
        Closes the viewer window.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            

def key_press(k, mod):
    global close
    if k == key.ESCAPE or k == key.X:
        close = True
        print('Closed')

def mouse_scroll(x, y, scroll_x, scroll_y):
    print(env.zoom)
    if scroll_y > 0 and env.zoom > 3:
        env.zoom_out()
    if scroll_y < 0 and env.zoom < 80:
        env.zoom_in()



def run_animation(data):
    """
    Runs complete animation 
    Parameters
    ---------
        env : CarEnv
            A car environment
        close : Boolean
            True if animation is finished or canceled
    """
    global env, close

    env = CarEnv()
    env.reset()
    env.render()
    
    env.viewer.window.on_mouse_scroll = mouse_scroll
    env.viewer.window.on_key_press = key_press

    # TODO: ZATVARANIE OKNA

    print(data.columns)

    for index, row in data.iterrows():
        if close is True:
            close = False
            break
        observation = env.step(row)
        env.render()
    env.close()
    print(env)
    
    return 


if __name__ == '__main__':
    env = CarEnv()
    env.reset()
    env.render()
    
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_mouse_scroll = mouse_scroll

    df = pd.read_csv('/Users/simonferancik/PycharmProjects/car-components-app-lukas/ML/data/full_track_mpc.csv')

    print(df.columns)
    df['x'] = df['x'] - df['x'][0]
    df['y'] = df['y'] - df['y'][0]

    for index, row in df.iterrows():
        if close is True:
            break
        observation = env.step(row)
        env.render()
    env.close()