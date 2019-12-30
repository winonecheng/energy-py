from energypy.envs.battery import Battery
from energypy.envs.gym import CartPoleEnv, PendulumEnv, MountainCarEnv, MountainCarContinuousEnv

register = {
    'battery': Battery,
    'cartpole-v0': CartPoleEnv,
    'pendulum-v0': PendulumEnv,
    'mountain-car-v0': MountainCarEnv,
    'mountain-car-continuous-v0': MountainCarContinuousEnv,
}


def make_env(env_id, **kwargs):
    """ grabs class from register and initializes """
    if env_id not in register.keys():
        raise ValueError(
            '{} not available - available envs are {}'.format(env_id, list(register.keys()))
        )
    return register[str(env_id)](**kwargs)
