from energypy.envs.battery import Battery
from energypy.envs.gym import CartPoleEnv, PendulumEnv, MountainCarEnv

register = {
    'battery': Battery,
    'cartpole-v0': CartPoleEnv,
    'pendulum-v0': PendulumEnv,
    'mountaincar-v0': MountainCarEnv,
}


def make_env(env_id, **kwargs):
    """ grabs class from register and initializes """
    if env_id not in register.keys():
        raise ValueError(
            '{} not available - available envs are {}'.format(env_id, list(register.keys()))
        )
    return register[str(env_id)](**kwargs)
