
from multiprocessing import Pool
from functools import partial

import energypy

class SingleProcessCollector():
    def __init__(self, policy):
        self.policy = policy

    def collect(self, episodes):
        env_id = 'cartpole-v0'
        return [episode(self.policy, env_id) for _ in range(episodes)]


class MultiProcessCollector():

    def __init__(self, policy, n_jobs=2):
        self.n_jobs = n_jobs
        self.policies = [policy for _ in range(n_jobs)]

    def collect(self, episodes):

        env_id = 'cartpole-v0'
        datasets = []

        with Pool(self.n_jobs, maxtasksperchild=32) as p:
            while len(datasets) < episodes:
                datasets.extend(p.map(partial(episode, env_id=env_id), self.policies))

        return datasets[:episodes]


class ParallelEnvCollector():

    def collect(self, episodes):
        pass


from collections import UserDict
import numpy as np


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


def episode(policy, env_id):
    """ runs full episode, returns lists of datasets """

    dataset = Dataset()

    env = energypy.make_env(env_id)
    obs = env.reset()
    done = False
    while not done:
        action = policy(obs, env)
        next_obs, reward, done, info = env.step(action)

        dataset.append(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done
        )

    return dataset

def random_policy(obs, env):
    return env.action_space.sample()


if __name__ == '__main__':
    import timeit
    num = 1024

    def test_collectors():
        start = timeit.default_timer()

        collectors = [
            SingleProcessCollector(random_policy),
            MultiProcessCollector(random_policy, n_jobs=8)
        ]

        for collector in collectors:
            data = collector.collect(num)
            assert len(data) == num
            stop = timeit.default_timer()
            print(stop - start)

            rews = [np.sum(d['reward']) for d in data]
            assert np.mean(rews) != rews[0]

    test_collectors()
