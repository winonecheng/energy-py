import energypy

import os


from collections import UserDict, UserList
import numpy as np

from energypy.interface.policies import make_policy


def episode(policy_params, env_id):
    """ runs full episode, returns lists of datasets """

    policy = make_policy(**policy_params)
    dataset = SingleEpisodeDataset()

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
            directory=np.array(policy_params['directory']).reshape(1, 1),
            env_id=np.array(policy_params['env_id']).reshape(1, 1),

        )
        obs = next_obs

    return dataset


class SingleProcessCollector():
    def __init__(self, policy_params, env_id):
        self.policy_params = policy_params
        self.env_id = env_id

    def collect(self, episodes):
        return MultiEpisodeDataset([
            episode(self.policy_params, self.env_id)
            for _ in range(episodes)
        ])


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


class SingleEpisodeDataset(UserDict):
    # TODO defaultdict with empty np array!

    def append(self, **kwargs):
        for k, v in kwargs.items():

            if k in self.keys():
                super().__setitem__(
                    k,
                    np.concatenate([self[k], v], axis=0)
                )
            else:
                super().__setitem__(k, v)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def save(self, path, key):
        np.save(path, self[key])


class MultiEpisodeDataset(UserList):
    def save(self, path, keys):
        for idx, d in enumerate(self.data):
            for k in keys:
                d.save(
                    os.path.join(path, k+'_'+str(idx)), k
                )

    def collapse(self):
        data = self
        from copy import deepcopy
        all_data = deepcopy(data[0])
        for d in data[1:]:
            for k, v in d.items():
                all_data[k] = np.concatenate(
                    [all_data[k], v]
                )
        return all_data


if __name__ == '__main__':
    env_id = 'mountain-car-continuous-v0'
    policy_params = {
        'policy_id': 'softmax',
        'env_id': env_id,
        'weights_dir': None
    }
    collector = SingleProcessCollector(policy_params, env_id)
    data = collector.collect(2)


