from energypy.interface.collectors import *

from energypy.common.memories.memory import calculate_returns


def calculate_returns_wrapper(data):
    discount = 0.98 #TODO
    for ep in data:
        ep['returns'] = calculate_returns(ep['reward'], discount)
    return data


class SingleProcessLabeller:
    def __init__(self, map_fctn):
        self.map_fctn = map_fctn

    def label(self, data):
        return self.map_fctn(data)
