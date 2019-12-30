import energypy

import os

import tensorflow as tf

    # separating out the collecting from fitting
    # optimizer & policy params not on same obj (cool!)

    #  initalize the policy inside the episode (inside the p.map)
    #  last possible moment

from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp

import timeit
from energypy.interface.collectors import *

def main():

    def calculate_returns_wrapper(data):
        discount = 0.9 # TODO
        for ep in data:
            ep['returns'] = calculate_returns(ep['reward'], discount)
        return data


    def log_prob_wrapper(data):
        out = []
        for ep in data:
            policy = make_policy(
                policy_id=ep['policy_id'][0][0], #TODO
                env_id=ep['env_id'][0][0],
                weights=ep['weights'][0][0]
            )
            out.append(policy.log_prob(ep))

        return out

    optimizer = tf.keras.optimizers.Adam(0.001)

    # REINFORCE

    def directory(*args):
        return os.path.join(
            os.environ['HOME'], 'energy-py', 'experiments',
            *args
        )

    num_collect = 6
    env_id = 'mountain-car-continuous-v0'

    weights = directory('run_0', 'policy_0')
    policy_params = {
        'policy_id': 'gaussian',
        'env_id': env_id,
    }
    pol = make_policy(**policy_params)
    pol.save(weights)

    from energypy.interface.labeller import collapse_episode_data
    from energypy.common.memories.memory import calculate_returns

    from energypy.interface.labeller import SingleProcessLabeller

    for step in range(1, 5):

        policy_params['weights'] = weights

        collector = MultiProcessCollector(
            policy_params, env_id, n_jobs=6
        )

        start = timeit.default_timer()
        data = collector.collect(num_collect)

        coll = collapse_episode_data(data)
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

        print(np.mean(coll['reward']), loss)

if __name__ == '__main__':
    main()
