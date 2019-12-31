import tensorflow as tf

from energypy.interface.collectors import *


def reward_only_labeller(data):
    return {
        'reward': data['reward']
    }




class SingleProcessLabeller:
    def __init__(self, map_fctn):
        self.map_fctn = map_fctn

    def label(self, data):
        return self.map_fctn(data)


if __name__ == '__main__':
    collector = SingleProcessCollector(random_policy)
    data = collector.collect(16)

    labeller = SingleProcessLabeller(map_fctn=reward_only_labeller)
    labelled_data = labeller.label(data)

    labeller = SingleProcessLabeller(map_fctn=TensorFlowValueFunction())
    labelled_data = labeller.label(data)
