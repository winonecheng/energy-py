import energypy

import os

from energypy.interface.collectors import collapse_episode_data, make_policy, MultiProcessCollector
from energypy.common.memories.memory import calculate_returns
import numpy as np

from energypy.interface.labeller import SingleProcessLabeller
import tensorflow as tf

import timeit


def calculate_returns_wrapper(data):
    discount = 0.9 # TODO
    for ep in data:
        ep['returns'] = calculate_returns(ep['reward'], discount)
    return data


class Directory():
    def __init__(self, *args):
        self.home = os.path.join(
            os.environ['HOME'], 'energy-py',
            *args
        )

    def __call__(self, *args):
        return os.path.join(self.home, *args)


if __name__ == '__main__':
    directory = Directory('experiments', 'mountaincar-reinforce')
    optimizer = tf.keras.optimizers.Adam(0.001)

    # REINFORCE
    num_collect = 16
    env_id = 'mountain-car-continuous-v0'
    # env_id = 'cartpole-v0'

    weights_dir = directory('run_0', 'policy_0')
    policy_params = {
        'policy_id': 'gaussian',
        'env_id': env_id,
    }
    pol = make_policy(**policy_params)
    pol.save(weights_dir)

    for step in range(1, 500):

        policy_params['weights_dir'] = weights_dir

        collector = MultiProcessCollector(
            policy_params, env_id, n_jobs=6
        )

        start = timeit.default_timer()
        data = collector.collect(num_collect)

        # TODO operates in place!!!!
        # coll = collapse_episode_data(data)
        stop = timeit.default_timer()
        #print(stop - start)

        return_labeller = SingleProcessLabeller(calculate_returns_wrapper)
        data = return_labeller.label(data)

        pol = make_policy(**policy_params)

        with tf.GradientTape() as tape:
            loss = pol.get_loss(data)
            gradients = tape.gradient(loss, pol.trainable_variables)

        optimizer.apply_gradients(zip(gradients, pol.trainable_variables))

        weights = directory('run_0', 'policy_{}'.format(step))
        print(weights)
        pol.save(weights)

        rews = [np.sum(d['reward']) for d in data]

        print('reward {} loss {} step {}'.format(np.mean(rews), loss, step))
