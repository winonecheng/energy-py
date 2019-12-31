import energypy
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
import os

import tensorflow as tf

def collapse_episode_data(data):
    all_data = data[0]
    for d in data[1:]:
        for k, v in d.items():
            all_data[k] = np.concatenate(
                [all_data[k], v]
            )
    return all_data


class RandomPolicy():
    # TODO kwargs
    def __init__(self, env_id, **kwargs):
        self.env = energypy.make_env(env_id)

    def __call__(self, obs):
        return self.env.action_space.sample()

    def save(self, filepath):
        pass


class ParameterizedPolicy():
    def __init__(self, env_id, weights_dir=None):
        self.env = energypy.make_env(env_id)

        self.net = tf.keras.Sequential([
            Dense(32, input_shape=self.env.observation_space.shape),
            Dense(16),
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
        data = collapse_episode_data(data)
        #  check all same policy :)
        data = self.log_prob(data)
        return - tf.reduce_mean(data['log_prob'] * data['returns'])

def make_policy(policy_id=None, **kwargs):
    register = {
        'gaussian': ParameterizedPolicy,
        'random': RandomPolicy
    }

    if policy_id not in register.keys():
        raise ValueError('{} not available - available envs are {}'.format(policy_id, list(register.keys())))

    return register[policy_id](**kwargs)
