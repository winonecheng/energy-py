"""
# energy-py 3.0 - maturing of the vision (sample efficiency)
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

collection on single core, mulit core, gpu

objects
- env - can this be built into tf
- Dataset
- State & Obs space & action space (multiagent by defn!)
# different random sampling!

# env.state_space
# env.state_space.sample()
# env.state_space.data[env.cursor, 'Charge [MWh]']
# env.state_space[env.cursor, 'Charge [MWh]']

fitter = Fitter(value_function)
fitter.fit(labelled_results, policy)

functions
- policy
- value fctn

simple (common) data struct == np.array

Value function == more concrete than a Labeller
"""

import numpy as np


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
        targets = value_function.forward(states)
        error = targets - labels
        # value_function.backward(error)
        return value_function

    q = update(q, states, labels)
