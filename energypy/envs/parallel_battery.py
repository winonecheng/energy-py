import tensorflow as tf
import numpy as np

from collections import UserDict


class TensorFlowInfo(UserDict):
    # concrete interface w/ tf
    def __setitem__(self, k, v):
        super().__setitem__(k, v)

    def __getitem__(self, k):
        try:
            k, cursor = k

        except ValueError:
            k = k
            cursor = None
        val = super().__getitem__(k)

        if cursor is not None:
            return tf.expand_dims(val[:, cursor, :], 1)
        else:
            return val


class BatteryTensorFlowEnv():
    # action = charging is positive, discharging is negative [MW]
    def __init__(
        self,
        agents=64,
        initial_charge=0,
        capacity=2,
        power=4,
        max_episode_length=2016
    ):
        # three agents, 9 episode horizon, 4 dim in obs
        self.obs = tf.random.uniform((3, 9, 4))
        self.agents = agents
        self.power = power
        self.max_episode_length = max_episode_length

        from energypy.common.spaces import StateSpace, ActionSpace
        from energypy.common.spaces import PrimitiveConfig as Prim
        # todo different for each agent
        self.initial_charge = tf.fill((self.agents, 1, 1), initial_charge)
        self.action_space = ActionSpace().from_primitives(
            Prim('Power [MW]', -self.power, power, 'continuous', None)
        )

    def reset(self):
        self.cursor = 0

        self.info = TensorFlowInfo()

        self.info['Initial charge [MWh]'] = self.initial_charge

        self.info['Price [$/MWh]'] = tf.random.uniform(shape=(self.agents, self.max_episode_length, 1))

        self.info['observation'] = tf.random.uniform(shape=(self.agents, self.max_episode_length, 1))

        return self.info['observation', self.cursor]

    def step(self, actions):
        actions = tf.reshape(actions, (self.agents, 1, 1)) / 12.0

        #  single timestep
        old_charge = self.info['Initial charge [MWh]', -1]
        old_charge = tf.cast(old_charge, tf.float32)

        capacity = 4.0
        eff = 0.9
        new_charge = tf.clip_by_value(
            tf.add(old_charge, actions), tf.zeros_like(old_charge), tf.fill(actions.shape, capacity)
        )

        #  hourly basis
        gross_power = (new_charge - old_charge) * 12

        discharging = tf.where(gross_power < 0, gross_power, tf.zeros_like(gross_power))

        loss = tf.abs(discharging * (1 - eff))
        charge = old_charge + (gross_power / 12)

        prices = tf.expand_dims(
            self.info['Price [$/MWh]'][:, -1, :], 1)

        reward = prices * charge

        self.info['Final charge [MWh]'] = charge
        self.info['Loss [MW]'] = loss

        if self.cursor < self.obs.shape[0]:
            self.cursor += 1
            done = tf.fill((self.agents, 1, 1), False)
        else:
            done = tf.fill((self.agents, 1, 1), True)

        #  add

        return self.obs[:, self.cursor, :], reward, done, self.info


if __name__ == '__main__':

    agents = 1024
    import energypy

    from energypy.interface.collector import episode, random_policy
    import time

    start = time.time()
    # single = [episode(random_policy, 'battery') for _ in range(agents)]
    end = time.time()
    print(end - start)

    def random_policy_vectorized(obs, env):
        lows, highs = [], []

        for dim in env.action_space.values():
            lows.append(dim.low)
            highs.append(dim.high)
        return np.random.uniform(lows, highs)

    # parallel_episode
    start = time.time()
    policy = random_policy_vectorized
    env = BatteryTensorFlowEnv(agents=agents, max_episode_length=2016)
    obs = env.reset()
    random_policy_vectorized(obs, env)
    done = tf.fill((env.agents, 1, 1), False)

    while not all(done):
        action = policy(obs, env)
        action = tf.cast(np.zeros((env.agents, 1, 1)), tf.float32)
        next_obs, reward, done, info = env.step(action)
    end = time.time()
    print(end - start)
