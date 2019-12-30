import tensorflow as tf

from energypy.interface.collectors import *


def reward_only_labeller(data):
    return {
        'reward': data['reward']
    }


def collapse_episode_data(data):
    all_data = data[0]
    for d in data[1:]:
        for k, v in d.items():
            all_data[k] = np.concatenate(
                [all_data[k], v]
            )
    return all_data


class SingleProcessLabeller:
    def __init__(self, map_fctn):
        self.map_fctn = map_fctn

    def label(self, data):
        return self.map_fctn(data)

class TensorFlowValueFunction:

    def value(self, obs):
        return tf.constant(0)

    def __call__(self, data):
        try:
            return {
                'value': self.value(data['obs'])
            }
        except TypeError:
            data = collapse_episode_data(data)
            return {
                'value': self.value(data['obs'])
            }

class TensorFlowFitter:
    def __init__(self, params):
        self.params = params

    def fit(self, data):
        take_gradient(data['loss'], self.params)


if __name__ == '__main__':
    collector = SingleProcessCollector(random_policy)
    data = collector.collect(16)

    labeller = SingleProcessLabeller(map_fctn=reward_only_labeller)
    labelled_data = labeller.label(data)

    labeller = SingleProcessLabeller(map_fctn=TensorFlowValueFunction())
    labelled_data = labeller.label(data)
