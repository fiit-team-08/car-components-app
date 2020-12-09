# https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
import gym
from gym import spaces
import pandas as pd
import numpy as np
import shapely.geometry as geom
from obspy.geodetics import degrees2kilometers
from RL.bicycle_model import BicycleKinematicModel


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
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, filename='../data/ref1_v2.csv', type='continuous',
                 action_dim=1, verbose=1):

        super(CarEnv, self).__init__()

        self.type = type
        self.action_dim = action_dim
        self.verbose = verbose
        self.df = self._read_df(filename)
        self.road = self._create_road()

        # 130 km/h ~= 36 m/s
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, 0., -(np.pi/4)]),
            high=np.array([np.inf, np.inf, np.pi, 36., np.pi/4])
        )

        # steering cmd a.k.a. psi and velocity
        # delta speed limit per 0.1s - 10km/h in 2s
        self.action_space = spaces.Box(
            low=np.array([-(np.pi/4), -0.14]),
            high=np.array([np.pi/4, 0.14])
        )

        self.viewer = None
        self.init_x = self.df['LON'][0]
        self.init_y = self.df['LAT'][0]
        self.init_theta = self.df['CRS'][0]
        self.init_v = 2.0
        self.init_psi = 0.0
        self.init_observation = np.array([self.init_x, self.init_y,
                                          self.init_theta, self.init_v,
                                          self.init_psi])
        self.state = None
        self.bicycle = None

    @staticmethod
    def _read_df(filename: str) -> pd.DataFrame:
        df = pd.read_csv(filename)
        # covert degrees to meters and degrees to radians
        df['LAT'] = df['LAT'].apply(lambda deg: degrees2kilometers(deg) * 1000)
        df['LON'] = df['LON'].apply(lambda deg: degrees2kilometers(deg) * 1000)
        df['CRS'] = df['CRS'].apply(lambda deg: np.deg2rad(deg))

        return df

    def _create_road(self):
        points = list()

        for index, row in self.df.iterrows():
            points.append(geom.Point(row['LON'], row['LAT']))

        road = geom.LineString(points)

        return road

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
        info = {'current_state': {'x': self.state[0],
                                  'y': self.state[1],
                                  'theta': self.state[2],
                                  'v': self.state[3],
                                  'psi': self.state[4]},
                'action': action}

        v = self.state[3] + action[1]
        delta_psi = action[0]

        self.bicycle.change_state(velocity=v, steering_rate=delta_psi)
        new_x, new_y, new_psi, new_theta = self.bicycle.get_state()
        new_point = geom.Point(new_x, new_y)
        distance = self.road.distance(new_point)

        if distance > 0.5:
            reward = -100
            done = True
        else:
            reward = (0.5 - distance) * 10.0

        self.state = [new_x, new_y, new_theta, v, new_psi]

        info['distance'] = distance
        info['reward'] = reward
        info['new_state'] = {'x': self.state[0],
                             'y': self.state[1],
                             'theta': self.state[2],
                             'v': self.state[3],
                             'psi': self.state[4]}

        # update info (include previous state, new state and reward)
        observation = self.state

        return observation, reward, done, info

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

        return self.init_observation

    def render(self, mode='human'):
        """
        Renders an animation to visualize the state of the car.
        """
        pass

    def close(self):
        """
        Closes the viewer window.
        """

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':

    env = CarEnv()
    for i_episode in range(10):
        observation = env.reset()
        for t in range(10000):
            # env.render()
            # with each action just turn the wheel +0.05 rad
            action = np.array([0.05, 0.0])
            observation, reward, done, info = env.step(action)
            print(info)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
