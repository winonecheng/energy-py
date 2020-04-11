import numpy as np

import energypy


class RandomAgent():
    def __init__(self, parameter_server, env):
        self.parameter_server = parameter_server

        if isinstance(env, str):
            env = energypy.make(env)
        self.env = env

    def act(self, obs):
        return np.array(self.env.observation_space.sample()).reshape(-1, 1)
