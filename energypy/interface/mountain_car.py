import energypy

import tensorflow as tf

from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp

import timeit
from energypy.interface.collectors import *


class ParameterizedPolicy():
    def __init__(self, env_id):
        print('new pol')
        self.env = energypy.make_env(env_id)

        self.net = tf.keras.Sequential([
            Dense(32, input_shape=self.env.observation_space.shape),
            Dense(16),
            # Dense(len(self.env.action_space.shape))
            Dense(1)
        ])

        self.load()

    def load(self):
        pass

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

    def log_prob(self, data):
        for episode in data:
            obs = episode['obs']
            action = episode['action']
            dist = tfp.distributions.Normal(self.net(obs), scale=1)
            episode['log_prob'] = dist.log_prob(action)
        return data

if __name__ == '__main__':

    # REINFORCE
    # env = energypy.make_env('mountain-car-continuous-v0')
    # obs = env.reset()
    # act = pol(obs)

    num_collect = 6

    # separating out the collecting from fitting
    # optimizer & policy params not on same obj (cool!)

    #  initalize the policy inside the episode (inside the p.map)
    #  last possible moment
    pol = ParameterizedPolicy('mountain-car-continuous-v0')
    collector = MultiProcessCollector(ParameterizedPolicy, 'mountain-car-continuous-v0', n_jobs=6)
    start = timeit.default_timer()
    data = collector.collect(num_collect)
    stop = timeit.default_timer()
    print(stop - start)

    from energypy.interface.labeller import collapse_episode_data
    from energypy.common.memories.memory import calculate_returns

    def calculate_returns_wrapper(data):
        discount = 0.9 # TODO
        for ep in data:
            ep['returns'] = calculate_returns(ep['reward'], discount)
        return data

    from energypy.interface.labeller import SingleProcessLabeller
    return_labeller = SingleProcessLabeller(calculate_returns_wrapper)
    data = return_labeller.label(data)

    def calculate_loss(data):
        # decorator !!
        for ep in data:
            ep['loss'] = - ep['log_prob'] * ep['returns']
        return data

    # log_probs = pol.log_prob(data)

    loss_labeller = SingleProcessLabeller(calculate_loss)
    data = loss_labeller.label(data)

    optimizer = tf.keras.optimizers.Adam(0.0001)

    # with tf.GradientTape() as tape:
    #     losses = data['loss']
    #     gradients = tape.gradient(sum(losses.values()), pol.trainable_variables)

    # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
