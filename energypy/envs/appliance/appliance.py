from random import random
import pkg_resources

import numpy as np
import pandas as pd

from energypy.common.spaces import StateSpace, ActionSpace
from energypy.common.spaces import PrimitiveConfig as Prim
from energypy.envs import BaseEnv


class Appliance(BaseEnv):
    def __init__(
            self,
            name,

            episode_length=2,
            sample_strat='fixed&full',

            dataset='data',
            **kwargs
    ):
        self.name = name
        self.sample_strat = sample_strat

        super().__init__(**kwargs)

        # TODO Fix highest value
        self.state_space = StateSpace().from_dataset(dataset).append(
                Prim('Power', 0, 10, 'continuous', 'append')
            )

        self.observation_space = self.state_space
        assert self.state_space.num_samples == self.observation_space.num_samples

        self.episode_length = min(episode_length, self.state_space.num_samples)

        # TODO Fix highest/lowest value
        self.action_space = ActionSpace().from_primitives(
            Prim('Up/Down', -1, 1, 'continuous', None)
        )

        # load init power data
        self.init_power_df = pd.read_csv(join(dataset, 'init_appliance.csv'), index_col=0, parse_dates=True, usecols=[name])

        # load tolerable power data
        self.tolerable_power_df = pd.read_csv(join(dataset, 'tolerable.csv'), index_col=0, parse_dates=True, usecols=[name])
    
    def __repr__(self):
        return f'<energypy APPLIANCE env - {self.name}>'

    def _reset(self):
        _episode = self.episode % (self.state_space.num_samples // self.episode_length)
        self.start = _episode * self.episode_length

        # TODO !!chech the index!!
        print(f'init_power index: {self.start}')
        self.power = self.init_power_df[self.start]

        self.state = self.state_space(
            self.steps, self.start, append={'Power': self.power}
        )
        self.observation = self.observation_space(
            self.steps, self.start, append={'Power': self.power}
        )

        assert self.power >= 0

        return self.observation

    def _step(self, action):
        
        old_power = self.power
        new_power = old_power + action

        # TODO !!chech the index!!
        print(f'tolerable_power index: {self.start + self.steps}')
        tolerable_power = self.tolerable_power_df.iloc[self.start + self.steps]

        # TODO add random to new_power in else condition
        self.power = new_power if new_power <= tolerable_power else tolerable_power

        # TODO fix reward function
        reward = tolerable_power - self.power

        #  zero indexing steps
        if self.steps == self.episode_length - 1:
            done = True
            next_state = np.zeros((1, *self.state_space.shape))
            next_observation = np.zeros((1, *self.observation_space.shape))

        else:
            done = False
            next_state = self.state_space(
                self.steps + 1, self.start,
                append={'Power': float(self.power)}
            )
            next_observation = self.observation_space(
                self.steps + 1, self.start,
                append={'Power': float(self.power)}
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

            'Initial power': old_power,
            'Final power': self.power,
            'Tolerable power': tolerable_power
        }