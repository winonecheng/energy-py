from envs import MountainCarWrapper
from agents import RandomAgent


env_registry = {
    'mountaincar': MountainCarWrapper
}

def make_env(env):
    return env_registry[env]

agent_registry = {
    'random': RandomAgent
}

def make_agent(agent):
    return agent_registry[agent]
