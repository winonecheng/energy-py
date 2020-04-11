import gym


class NoSpace:
    def sample(self):
        return 0


class NoEnv():
    def __init__(self):
        self.observation_space = NoSpace()
        self.action_space = NoSpace()

    def step(self, action):
        return 0, 0, True, {}

    def reset(self):
        return 0

    def seed(self, seed):
        pass


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
