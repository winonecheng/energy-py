import tensorflow as tf
import numpy as np


# action = charging is positive, discharging is negative [MW]

class BatteryTensorFlowEnv():
    def __init__(
        self,
        agents=64,
        initial_charge=0,
        capacity=2,
    ):
        # three agents, 9 episode horizon, 4 dim in obs
        self.obs = tf.random.uniform((3, 9, 4))
        self.agents = agents
        max_episode_length = 32
        self.max_episode_length = max_episode_length

        # todo different for each agent
        self.initial_charge = tf.fill((self.agents, 1, 1), initial_charge)

    def reset(self):
        self.cursor = 0

        from collections import defaultdict
        self.info = defaultdict(list)

        self.info['Initial charge [MWh]'] = self.initial_charge

        self.info['Price [$/MWh]'] = tf.random.uniform(shape=(env.agents, self.max_episode_length, 1))

        self.info['observation'] = tf.random.uniform(shape=(env.agents, self.max_episode_length, 1))

        return self.info['observation'][:, self.cursor, :]

    def step(self, actions):
        actions = tf.reshape(actions, (self.agents, 1, 1)) / 12

        old_charge = self.info['Initial charge [MWh]'][:, -1, :]
        #  single timestep
        old_charge = tf.reshape(old_charge, (self.agents, 1, 1))

        capacity = 4.0
        eff = 0.9
        new_charge = tf.clip_by_value(
            tf.add(old_charge, actions), tf.zeros_like(old_charge), tf.fill(actions.shape, capacity)
        )

        #  hourly basis
        gross_power = new_charge - old_charge

        discharging = tf.where(gross_power < 0, gross_power, tf.zeros_like(gross_power))

        loss = tf.abs(discharging * (1 - eff))
        charge = old_charge + gross_power

        prices = tf.reshape(actions, (self.agents, 1))
        prices = self.info['Price [$/MWh]'][:, -1, :]

        reward = prices * charge
        done = tf.fill((3, 1), False)

        try:
            self.info['Final charge [MWh]'] = tf.concat([
                self.info['Final charge [MWh]'],
                charge], axis=1
            )
        except tf.errors.InvalidArgumentError:
            self.info['Final charge [MWh]'] = charge

        prices = tf.constant([0.5, 4.0, -0.5])

        self.cursor += 1
        return self.obs[:, self.cursor, :], reward, done, self.info

# dataset = Example()

def test_charge():
    cfg = {
        'agents': 4,
    }

    env = BatteryTensorFlowEnv(**cfg)

    obs = env.reset()
    action = tf.fill((env.agents, 1), 1.0)
    next_ob, reward, done, info = env.step(action)

    np.testing.assert_allclose(info['Final charge [MWh]'][:, -1, :], np.full(cfg['agents'], 1.0 / 12).reshape(env.agents, 1))


# def test_discharge():

# def test no nop
cfg = {
    'agents': 4,
    'initial_charge': 1.0,
    'capacity': 4.0
}

env = BatteryTensorFlowEnv(**cfg)
obs = env.reset()

action = tf.fill((env.agents, 1), -1.0)
rew, next_obs, d, info = env.step(action)

charge = info['Final charge [MWh]'][:, -1, :]
assert charge == np.full((cfg['agents']), 4.0 - 1.0 / 12).reshape(env.agents, 1)

losses = info['Loss [MW]'][:, -1, :]
assert losses == np.full((cfg['agents'], 1.0 * (1 - 0.9))).reshape(env.agents, 1)
