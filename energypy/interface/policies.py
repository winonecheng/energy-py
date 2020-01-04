import energypy
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
import os

import numpy as np

import tensorflow as tf


def make_policy(policy_id=None, **kwargs):
    register = {
        'gaussian': GaussianPolicy,
        'random': RandomPolicy,
        'softmax': SoftmaxPolicy,
    }

    if policy_id not in register.keys():
        raise ValueError('{} not available - available envs are {}'.format(policy_id, list(register.keys())))

    return register[policy_id](**kwargs)


class RandomPolicy():
    def __init__(self, env_id, **kwargs):
        self.env = energypy.make_env(env_id)

    def __call__(self, obs):
        return self.env.action_space.sample()

    def save(self, filepath):
        pass


class GaussianPolicy():
    def __init__(self, env_id, weights_dir=None):
        self.env = energypy.make_env(env_id)

        self.net = tf.keras.Sequential([
            Dense(4, input_shape=self.env.observation_space.shape),
            Dense(len(self.env.action_space))
        ])

        self.trainable_variables = self.net.trainable_variables

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
        data = data.collapse()
        #  check all same policy :)
        data = self.log_prob(data)
        return - tf.reduce_mean(data['log_prob'] * data['returns'])


class SoftmaxPolicy():
    def __init__(self, env_id, directory=None):
        self.env = energypy.make_env(env_id)

        discrete_actions = self.env.action_space.discretize(20)

        self.net = tf.keras.Sequential([
            Dense(128, input_shape=self.env.observation_space.shape),
            Dense(len(discrete_actions), activation='softmax')
        ])

        self.trainable_variables = self.net.trainable_variables

        if directory:
            self.load(directory)

    def load(self, filepath):
        #print('loading model from {}'.format(filepath))
        self.net.load_weights(filepath + '/weights.h5')

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        #print('saving model to {}'.format(filepath))
        self.net.save_weights(os.path.join(filepath, 'weights.h5'))

    def __call__(self, obs):
        # single action only
        probs = self.net(obs)
        return tf.random.categorical(probs, obs.shape[0]).numpy().reshape(1, 1).astype(np.int)

    def log_prob(self, data):
        # probability of the action actually taken!
        assert data['action'].shape[1] == 1
        data['log_prob'] = tf.reduce_sum(
            tf.one_hot(data['action'].flatten(), 2) * tf.math.log(self.net(data['obs'])),
            axis=1, keepdims=True)
        return data

    def get_loss(self, data):
        data = data.collapse()
        data = self.log_prob(data)
        return - tf.reduce_mean(data['log_prob'] * data['returns'])
