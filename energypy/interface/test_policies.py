

import energypy

from energypy.interface.policies import make_policy

# def test_softmax_log_prob
env_id = 'cartpole-v0'
policy_params = {
    'policy_id': 'softmax',
    'env_id': env_id,
    'directory': None
}
policy = make_policy(**policy_params)

env = energypy.make_env(env_id)
obs = env.reset()
done = False

from collectors import SingleEpisodeDataset, MultiEpisodeDataset

dataset = SingleEpisodeDataset()

import numpy as np
steps = 3

for st in range(steps):
    action = policy(obs)
    next_obs, reward, done, info = env.step(action)

    dataset.append(
        obs=obs,
        action=action,
        reward=reward
    )
    obs = next_obs

from energypy.interface.labeller import SingleProcessLabeller
from energypy.interface.mountain_car import calculate_returns_wrapper
return_labeller = SingleProcessLabeller(calculate_returns_wrapper)
data = return_labeller.label((dataset,))[0]

obss = dataset['obs']
acts = dataset['action']

pol_probs = policy.log_prob(data)
test_probs = np.log(policy.net(obss))

for st in range(steps):

    act = acts[st, :]
    test_prob = test_probs[st, act]

    pol_prob = np.sum(pol_probs['log_prob'][st, :], keepdims=True)
    np.testing.assert_array_equal(test_prob, pol_prob)

# test the loss
pol_loss = policy.get_loss(MultiEpisodeDataset((dataset,)))
test_loss = - np.mean(pol_probs['log_prob'] * pol_probs['returns'])
np.testing.assert_array_equal(pol_loss, test_loss)
