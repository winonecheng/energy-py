import numpy as np
import rethink as energypy


class RandomAgent():
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return np.array(self.env.observation_space.sample()).reshape(-1, 1)

    def learn(self):
        pass


def test_random_agent():
    env = energypy.make_env('mountaincar')
    agent = RandomAgent(env)
    agent.act


if __name__ == '__main__':
    test_random_agent()
