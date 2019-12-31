from multiprocessing import Pool
from functools import partial

import energypy


from collections import UserDict
import numpy as np

from energypy.interface.policies import make_policy


def episode(policy_params, env_id):
    """ runs full episode, returns lists of datasets """

    policy = make_policy(**policy_params)
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
            done=done,
            policy_id=np.array(policy_params['policy_id']).reshape(1, 1),
            weights_dir=np.array(policy_params['weights_dir']).reshape(1, 1),
            env_id=np.array(policy_params['env_id']).reshape(1, 1),

        )

    return dataset



class SingleProcessCollector():
    def __init__(self, policy_params, env_id):
        self.policy_params = policy_params
        self.env_id = env_id

    def collect(self, episodes):
        return [episode(self.policy_params, self.env_id) for _ in range(episodes)]


class MultiProcessCollector():

    def __init__(self, policy_params, env_id, n_jobs):
        self.n_jobs = n_jobs
        self.env_id = env_id
        self.policies = [policy_params for _ in range(n_jobs)]

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
