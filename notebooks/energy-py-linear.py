#!/usr/bin/env python

import numpy as np

from energypy.experiments.blocks import perform_episode

import energypy as ep
import energypylinear as epl

from energypy.common import make_memory

from energypy.agents import BaseAgent

class FixedAgent(BaseAgent):
    def __init__(self, actions, **kwargs):
        self.actions = actions
        self.cursor = -1

        super().__init__(memory_length=len(self.actions), **kwargs)

    def _act(self, obs):
        if self.cursor == len(self.actions) - 1:
            self.cursor = 0
        else:
            self.cursor += 1
        return self.actions[self.cursor]

    def _reset(self):
        self.cursor = -1

    def _learn(self):
        pass

np.random.seed(42)
prices = np.random.rand(5) * 100

import pandas as pd
trading = pd.read_csv('/Users/adam/GoogleDrive/clean-aemo-data/clean/trading_price.csv', index_col=0, parse_dates=True)
prices = trading.loc[:, 'RRP SA1'].values

# prices = prices[:10000]

linear = epl.Battery(power=2, capacity=4, efficiency=0.9)

batt = ep.make_env(
    'battery',
    power=2,
    capacity=4,
    sample_strat='full',
    prices=prices,
    efficiency=0.9
)

opt = linear.optimize(prices, initial_charge=batt.initial_charge)
actions = [d['Gross [MW]'] for d in opt]
agent = FixedAgent(actions, env=batt)

agent.reset()

assert len(agent.actions) == len(prices)
info = perform_episode(agent, batt)

assert actions == info['action']

opt_rewards = [-d['Actual [$/5min]'] for d in opt]

print(sum(info['reward']))
print(sum(opt_rewards))

#  we have a memory full of fixed agent actions
#  lets learn a q function

import tensorflow as tf
with tf.Session() as sess:
    dqn = ep.make_agent(
        'dqn',
        env=batt,
        sess=sess,
        tensorboard_dir='/Users/adam/energy-py-results/pretrain'
    )
    dqn.memory = agent.memory
    epochs = 10

    batches = epochs * len(dqn.memory)
    print('learning for {} batches'.format(batches))

    for _ in range(batches):
        dqn.learn()


