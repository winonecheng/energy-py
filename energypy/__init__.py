from energypy.agents import RandomAgent
from energypy.envs import MountainCarWrapper, NoEnv
from energypy.learners import AddLearner


registry = {
    'add-learner': AddLearner,

    'noenv': NoEnv,
    'no-env': NoEnv,

    'mountaincar': MountainCarWrapper,
    'mountain-car': MountainCarWrapper,
    'random': RandomAgent
}


def make(name, *args, **kwargs):
    return registry[name](*args, **kwargs)
