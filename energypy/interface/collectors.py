from multiprocessing import Pool
from functools import partial

import energypy

import os

import tensorflow as tf
from energypy.interface.labeller import collapse_episode_data

from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
from collections import UserDict
import numpy as np

class ParameterizedPolicy():
    def __init__(self, env_id, weights=None):
        self.env = energypy.make_env(env_id)

        self.net = tf.keras.Sequential([
            Dense(32, input_shape=self.env.observation_space.shape),
            Dense(16),
            # Dense(len(self.env.action_space.shape))
            Dense(1)
        ])

        self.trainable_variables = self.net.trainable_variables

        if weights == 'latest':
            direct = directory()
            self.load(weights)

    def load(self, filepath):
        print('loading model from {}'.format(filepath))
        self.net.load_weights(filepath)

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        print('saving model to {}'.format(filepath))
        self.net.save_weights(os.path.join(filepath, 'weights.h5'))

    def __call__(self, obs):
        # single action
        action = tfp.distributions.Normal(self.net(obs), scale=1)
        action = action.sample()

        import numpy as np
        return tf.clip_by_value(
            action,
            self.env.action_space.low,
            self.env.action_space.high
        ).numpy().reshape(1, 1).astype(np.float32)

    def log_prob(self, episode):
        obs = episode['obs']
        action = episode['action']
        dist = tfp.distributions.Normal(self.net(obs), scale=1)
        episode['log_prob'] = dist.log_prob(action)
        return episode

    def get_loss(self, data):
        data = collapse_episode_data(data)
        #  check all same policy :)
        data = self.log_prob(data)

        return - tf.reduce_mean(data['log_prob'] * data['returns'])

def make_policy(policy_id=None, **kwargs):
    register = {
        'gaussian': ParameterizedPolicy
    }

    if policy_id not in register.keys():
        raise ValueError('{} not available - available envs are {}'.format(policy_id, list(register.keys())))

    return register[policy_id](**kwargs)


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
            weights=np.array(policy_params['weights']).reshape(1, 1),
            env_id=np.array(policy_params['env_id']).reshape(1, 1),

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
