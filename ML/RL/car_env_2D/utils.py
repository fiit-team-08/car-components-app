import numpy as np


class Simulator:
    def __init__(self, x, y, heading_angle, steering_angle, bicycle_len=1,
                 dt=0.1):
        self.x = x                              # meters
        self.y = y                              # meters
        # theta
        self.heading_angle = heading_angle      # radians
        # delta
        self.steering_angle = steering_angle    # radians

        self.bicycle_len = bicycle_len          # meters
        self.dt = dt                            # seconds
        self.last_state = None

    def get_state(self):
        return self.x, self.y, self.heading_angle, self.steering_angle

    def change_state(self, velocity, steering_rate):
        # velocity          m/s
        # steering_rate     radians

        self.last_state = [self.x, self.y, self.heading_angle, self.steering_angle]

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
