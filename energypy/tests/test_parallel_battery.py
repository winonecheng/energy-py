
from energypy.envs.parallel_battery import *


def test_charge():
    cfg = {
        'agents': 4,
    }

    env = BatteryTensorFlowEnv(**cfg)

    obs = env.reset()
    action = tf.fill((env.agents, 1), 1.0)
    next_ob, reward, done, info = env.step(action)

    np.testing.assert_allclose(info['Final charge [MWh]'][:, -1, :], np.full(cfg['agents'], 1.0 / 12).reshape(env.agents, 1))


def test_discharge():

    cfg = {
        'agents': 4,
        'initial_charge': 4.0,
        'capacity': 4.0
    }

    env = BatteryTensorFlowEnv(**cfg)
    obs = env.reset()

    action = tf.fill((env.agents, 1, 1), -1.0)
    next_obs, rew, d, info = env.step(action)

    charge = info['Final charge [MWh]'][:, -1, :]
    np.testing.assert_array_almost_equal(charge, np.full((cfg['agents']), 4.0 - 1.0 / 12).reshape(env.agents, 1))

    losses = info['Loss [MW]'][:, -1, :]
    np.testing.assert_array_almost_equal(losses, np.full((cfg['agents']), 1.0 * (1 - 0.9)).reshape(env.agents, 1))



def test_tf_info():
    data = tf.random.uniform(shape=(2, 4, 8))
    inf = TensorFlowInfo()
    inf['test'] = data

    out0 = inf['test', 0]
    np.testing.assert_array_almost_equal(out0, data[:, 0, :].numpy().reshape(2, 1, 8))

    out1 = inf['test', 1]
    np.testing.assert_array_almost_equal(out1, data[:, 1, :].numpy().reshape(2, 1, 8))

    out = inf['test']
    np.testing.assert_array_almost_equal(out, data)
