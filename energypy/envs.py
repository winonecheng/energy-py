
def test_mountain_car_wrapper():
    env = MountainCarWrapper()
    env.seed(42)
    obs = env.reset()
    obs, r, done, info = env.step(env.action_space.sample())

import gym


class MountainCarWrapper():
    def __init__(self):
        self.env = gym.make('MountainCarContinuous-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def __repr__(self):
        return '<wrapper of {}'.format(self.env.__repr__())

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        return obs.reshape(1, -1), r, done, info

    def reset(self):
        return self.env.reset().reshape(1, -1)

    def seed(self, seed):
        return self.env.seed(seed)


if __name__ == '__main__':
    test_mountain_car_wrapper()

