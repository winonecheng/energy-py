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

            episode_length=720,
            sample_strat='full',

            dataset='data',
            **kwargs
    ):
        self.name = name
        self.sample_strat = sample_strat

        super().__init__(**kwargs)

        self.state_space = StateSpace().from_dataset(dataset)

        self.observation_space = self.state_space
        assert self.state_space.num_samples == self.observation_space.num_samples

        if sample_strat == 'full':
            self.episode_length = self.state_space.num_samples
        else:
            self.episode_length = min(episode_length, self.state_space.num_samples)

        self.action_space = ActionSpace().from_primitives(
            Prim('Turn down', -0.1, 0, 'continuous', None)
        )

        # load tolerable power data
        self.tolerable_power_df = pd.read_csv(join(dataset, 'tolerable.csv'), parse_dates=True)[name]
    
    def __repr__(self):
        return f'<energypy APPLIANCE env - {self.name}>'

    def _reset(self):
        self.start, self.end = self.state_space.sample_episode(
            self.sample_strat, episode_length=self.episode_length
        )

        # TODO check!!
        self.state = self.state_space(
            self.steps, self.start
        )
        self.observation = self.observation_space(
            self.steps, self.start
        )

        # TODO check!!
        self.power = self.get_state_variable(self.name)

        assert self.power >= 0

        return self.observation

    def _step(self, action):
        # TODO check!!
        old_power = self.power
        _new_power = old_power + action[0][0]

        tolerable_power = self.tolerable_power_df.iloc[self.steps]

        self.power = _new_power if _new_power >= tolerable_power else tolerable_power

        reward = self.power - old_power

        # print(f'old_power: {old_power}, action: {action}, tolerable_power: {tolerable_power}, new_power: {self.power}, reward:{reward}')

        #  zero indexing steps
        if self.steps == self.episode_length - 1:
            done = True
            next_state = np.zeros((1, *self.state_space.shape))
            next_observation = np.zeros((1, *self.observation_space.shape))

        # TODO check!!
        else:
            done = False
            next_state = self.state_space(
                self.steps + 1, self.start
            )
            next_observation = self.observation_space(
                self.steps + 1, self.start
            )
            # print(next_state)

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
            'Tolerable power': tolerable_power
        }