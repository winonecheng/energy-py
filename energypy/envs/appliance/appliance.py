from random import random
from os.path import join

import numpy as np
import pandas as pd

from energypy.common.spaces import StateSpace, ActionSpace
from energypy.common.spaces import PrimitiveConfig as Prim
from energypy.envs import BaseEnv


class Appliance(BaseEnv):
    def __init__(
            self,
            name,

            episode_length=24, # one day 24 hr

            dataset='data',
            **kwargs
    ):
        self.name = name
        self.episodes = 0
        self.episode_length = episode_length

        super().__init__(**kwargs)

        self.state_space = StateSpace().from_dataset(dataset)
        self.observation_space = self.state_space
        assert self.state_space.num_samples == self.observation_space.num_samples

        self.action_space = ActionSpace().from_primitives(
            Prim('Turn down', -1, 1, 'continuous', None)
        )

        # load tolerable power data
        self.tolerable_power_df = pd.read_csv(join(dataset, 'tolerable.csv'), parse_dates=True)[name]
    
    def __repr__(self):
        return f'<energypy APPLIANCE env - {self.name}>'

    def _reset(self):
        self.start = self.episodes * self.episode_length

        self.state = self.state_space(
            self.steps, self.start
        )
        self.observation = self.observation_space(
            self.steps, self.start
        )

        return self.observation

    def _step(self, action):
        _action = (action - 1) * 0.1 # range [-1, 1] -> [-0.2, 0]
        old_power = self.get_state_variable(self.name)
        _new_power = old_power + _action
        tolerable_power = self.tolerable_power_df.iloc[self.start + self.steps]

        # user do feedback
        if _new_power < tolerable_power:
            reward = _new_power - tolerable_power
            self.power = tolerable_power
        else:
            reward = old_power - _new_power
            self.power = _new_power


        # test: close to tolerable
        reward = 0 - abs(_new_power-tolerable_power)
        reward *= 10

        #  zero indexing steps
        if self.steps == self.episode_length - 1:
            done = True
            self.episodes += 1
        else:
            done = False

        next_state = self.state_space(
            self.steps + 1, self.start
        )
        next_observation = self.observation_space(
            self.steps + 1, self.start
        )

        return {
            'step': int(self.steps),
            'state': self.state,
            'observation': self.observation,
            'action': action,
            'reward': float(reward),
            'next_state': next_state,
            'next_observation': next_observation,
            'done': bool(done),

            'Old power': old_power,
            'Final power': self.power,
            'Tolerable power': tolerable_power,
            'Acturl action': _action
        }