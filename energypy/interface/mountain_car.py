"""
TODO
- multi episode dataset
- continue from last policy !!!
"""








import energypy

import os

from energypy.interface.policies import make_policy

from energypy.interface.collectors import MultiProcessCollector, SingleProcessCollector
import numpy as np

from energypy.interface.labeller import SingleProcessLabeller
import tensorflow as tf

import timeit


class CustomLogger():
    def __init__(self, log_file, header=None):
        self.log_file = os.path.join(log_file)

        if header:
            self(header)

    def __call__(self, msg, verbose=True):
        with open(self.log_file, 'a') as lf:
            lf.write(msg+'\n')
        if verbose:
            print(msg)


from energypy.interface.labeller import calculate_returns_wrapper

class Directory():
    def __init__(self, *args, delete=False):
        self.home = os.path.join(
            os.environ['HOME'], 'energy-py',
            *args
        )

        if delete:
            import shutil
            shutil.rmtree(self.home, ignore_errors=True)

    def __call__(self, *args):
        path = os.path.join(self.home, *args)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


if __name__ == '__main__':
    env_id = 'cartpole-v0'
    run_directory = Directory('experiments', env_id, 'run_0', delete=True)

    step = 0
    policy_params = {
        'policy_id': 'softmax',
        'env_id': env_id,
        'directory': None
    }
    pol = make_policy(**policy_params)

    init_pol = run_directory('policy_{}'.format(step))
    pol.save(init_pol)

    from collections import deque
    last_100 = deque(maxlen=100)

    optimizer = tf.keras.optimizers.Adam(0.01)
    for step in range(1, 10000):

        old_pol_dir = run_directory('policy_{}'.format(step - 1))
        policy_params['directory'] = old_pol_dir

        # collection
        collector = SingleProcessCollector(policy_params, env_id)
        data = collector.collect(1)
        data.save(old_pol_dir, keys=['reward'])

        # labelling
        return_labeller = SingleProcessLabeller(calculate_returns_wrapper)
        data = return_labeller.label(data)

        # fitting
        pol = make_policy(**policy_params)
        for _ in range(1):
            with tf.GradientTape() as tape:
                loss = pol.get_loss(data)
                gradients = tape.gradient(loss, pol.trainable_variables)

            optimizer.apply_gradients(zip(gradients, pol.trainable_variables))

        new_pol_dir = run_directory('policy_{}'.format(step))
        pol.save(new_pol_dir)

        # logging
        rews = [np.sum(d['reward']) for d in data]
        last_100.append(np.mean(rews))

        msg = 'last 100 avg {:3.1f} - last 100 max {:3.1f} - loss {:3.1f} step {:5.0f}'.format(
            np.mean(last_100),
            np.max(last_100),
            loss,
            step
        )
        logger = CustomLogger(run_directory('log.log'))
        logger(msg)
