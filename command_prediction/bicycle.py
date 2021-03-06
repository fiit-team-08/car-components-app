from math import cos, sin, tan, atan2
import numpy as np

class BicycleKinematicModel:

    def __init__(self, x, y, steering_angle, heading_angle, bicycle_len=1,
                 time_step=0.1):
        self._x = x  # meters
        self._y = y  # meters
        self._steering_angle = steering_angle  # radians
        self._heading_angle = heading_angle  # radians
        self.bicycle_len = bicycle_len  # meters
        self.time_step = time_step  # seconds

    def get_state(self):
        return self._x, self._y, self._steering_angle, self._heading_angle

    def change_state(self, velocity, steering_rate):
        """
            Changes its state according to the input: a velocity and a steering rate.
            The next state is evaluated regarding the reference point at the rear axle.
            Parameters
            ===========
                velocity : float
                    Current velocity of a bicycle.
                steering_rate : float
                    Angle change rate.
        """

        if self._steering_angle == 0:
            rotation_rate = 0

        else:
            R = self.bicycle_len / tan(self._steering_angle)
            rotation_rate = velocity / R

        delta_x = velocity * cos(self._heading_angle)
        delta_y = velocity * sin(self._heading_angle)
        delta_theta = rotation_rate
        delta_sigma = steering_rate

        self._x += delta_x * self.time_step
        self._y += delta_y * self.time_step
        self._heading_angle += delta_theta * self.time_step
        self._heading_angle = np.deg2rad(np.rad2deg(atan2(sin(self._heading_angle), cos(self._heading_angle))) % 360)
        self._steering_angle += delta_sigma
