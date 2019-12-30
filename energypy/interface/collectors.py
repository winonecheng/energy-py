from multiprocessing import Pool
from functools import partial

import energypy

from collections import UserDict
import numpy as np


def episode(policy, env_id):
    """ runs full episode, returns lists of datasets """
    policy = policy(env_id)
    dataset = Dataset()

    env = energypy.make_env(env_id)
    obs = env.reset()
    done = False
    while not done:
        action = policy(obs)
        next_obs, reward, done, info = env.step(action)

        dataset.append(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done
        )

    return dataset


class RandomPolicy():
    def __init__(self, env_id):
        self.env = energypy.make_env(env_id)

    def __call__(self, obs):
        return self.env.action_space.sample()


class SingleProcessCollector():
    def __init__(self, policy, env_id):
        self.policy = policy
        self.env_id = env_id

    def collect(self, episodes):
        return [episode(self.policy, self.env_id) for _ in range(episodes)]


class MultiProcessCollector():

    def __init__(self, policy, env_id, n_jobs):
        self.n_jobs = n_jobs
        self.env_id = env_id
        self.policies = [policy for _ in range(n_jobs)]

    def collect(self, episodes):
        datasets = []

        with Pool(self.n_jobs, maxtasksperchild=32) as p:
            while len(datasets) < episodes:
                datasets.extend(
                    p.map(partial(
                        episode,
                        env_id=self.env_id
                    ), self.policies)
                )

        return datasets[:episodes]


class ParallelEnvCollector():
    # tensorflow

    def collect(self, episodes):
        pass


class Dataset(UserDict):
    def append(self, **kwargs):
        for k, v in kwargs.items():

            if k in self.keys():
                super().__setitem__(
                    k,
                    np.concatenate([self[k], v], axis=0)
                )
            else:
                super().__setitem__(k, v)
