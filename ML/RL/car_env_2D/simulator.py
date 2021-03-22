import numpy as np


class Simulator:
    def __init__(self, x, y, heading_angle, steering_angle, bicycle_len=1,
                 dt=0.1):
        """
        Class for simulation of the car movement (Bicycle Kinematic Model).

        Parameters
        ----------
        x (float): X coordinate of the vehicle position in meters.
        y (float): Y coordinate of the vehicle position in meters.
        heading_angle (float): Heading angle af the vehicle in radians.
        steering_angle (float): Steering angle af the vehicle in radians.
        bicycle_len (int): Length of the vehicle in meters.
        dt: Delta of the time.
        """

        self.x = x                              # meters
        self.y = y                              # meters
        self.heading_angle = heading_angle      # radians (theta)
        self.steering_angle = steering_angle    # radians (delta)

        self.bicycle_len = bicycle_len          # meters
        self.dt = dt                            # seconds

    def get_state(self):
        return self.x, self.y, self.heading_angle, self.steering_angle

    def change_state(self, velocity, steering_rate):
        """
        Change the state according to the input (velocity and steering rate).
        The next state is evaluated regarding the reference point
        at the rear axle.

        Parameters
        ----------
        velocity (float): Current velocity of the vehicle im m/s.
        steering_rate (float): Angle change rate in radians.
        """

        x_dot = velocity * np.cos(self.heading_angle)
        y_dot = velocity * np.sin(self.heading_angle)
        theta_dot = (velocity * np.tan(self.steering_angle)) / self.bicycle_len
        delta_dot = steering_rate

        self.x += x_dot * self.dt
        self.y += y_dot * self.dt
        self.heading_angle += theta_dot * self.dt
        # self.steering_angle += delta_dot * self.dt
        self.steering_angle += delta_dot

        return self.x, self.y, self.heading_angle, self.steering_angle
