"""
Review minimal & huskarl

RL
- clean architecture

Benefit of smaller frameworks is to innovate (and maybe fail)

Rl is
- data collection
- data labelling
- fitting functions

Three interfaces

collector = Collector()

labeller = Labeller()

fitter = Fitter()
All envs are parallelized by default

"""

import numpy as np

class Env:
    def __init__(self, states):
        self.states = states

    def __call__(self, actions):
        #  return transition
        #  could return a tensor
        return (
            np.zeros((actions.shape[0], self.states.shape[1])),
            np.ones_like(actions),
            np.full((actions.shape[0], 1), -3),
            np.zeros((actions.shape[0], self.states.shape[1])),
            np.full((actions.shape[0], 1), False),
        )

def random_policy(states, action_space):
    #  vectorize it! 
    #  think about parallelization from the start!

    num_samples = states.shape[0]

    idxs = [np.random.randint(0, action_space.shape[0], size=(states.shape[0]))]

    tiled = np.tile(action_space, states.shape[0])
    #  tiles.shape = (num_samples, num_actions, action_shape)
    tiles = np.tile(action_space, (num_samples, 1, 1))
    # idx = np.random.randint(low=0, high=num_actions, size=(num_samples))
    # actions = tiles[:, idx, :]

    mask = np.random.choice(num_actions, num_samples, p=[1/num_actions for _ in range(num_actions)])

    actions = action_space[mask]

    print(actions.shape)
    assert actions.shape == (num_samples, 2)
    return actions

class Dataset(list):
    def __init__(self):
        pass

    def __getitem__(self, key):
        return list.__getitem__(self, key)

    def __setitem__(self, index, value):
        self._inner_list.__setitem__(index, value)

    def __call__(self, batch_size):
        return self[0:2]

if __name__ == '__main__':

    #  test random policy - vectorized
    # class Collector:

    # three actions of dim 2
    action_space = np.array([[0, 0], [1, 1], [2, 2]])
    num_actions = action_space.shape[0]

    #  10 samples of dim 3
    states = np.random.rand(10, 3)

    actions = random_policy(states, action_space)

    env = Env(states)
    samples = env(actions)

    dataset = Dataset()
    dataset.append(samples)
    dataset.append(samples)

    batch = dataset(64)
    names = ['state', 'action', 'reward', 'next-state']

    print([(name, data) for name, data in zip(names, batch[0])])

    class ValueFunction:
        def __init__(self):
            pass

        def __call__(self, states):
            return 1

        def get_actions(self, states, actions):
            return actions

    q = ValueFunction()

    # class Labeller
    def bellman(value_function, batch):
        rewards = (batch[0][2])
        next_state_values = value_function(batch[0][3])
        return rewards + next_state_values

    labels = bellman(q, batch)

    # class Fitter
    def update(value_function, states, labels):
        #  maybe could take state as input
        #  features, label
        targets = value_function(states)
        error = targets - labels
        # value_function.backward(error)
        return value_function

    q = update(q, states, labels)
