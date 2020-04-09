import rethink as energypy

import numpy as np

import json
import os
import pathlib

import ray


class File:
    def __init__(self, home ):
        self.home = pathlib.Path(
            os.path.expanduser('~'),
            'energypy',
            home
        )
        self.home.parent.mkdir(parents=True, exist_ok=True)

    def save(self, episodes):
        for transitions in episodes:
            # TODO ray dep
            transitions = [
                {k: np.array(v).flatten().tolist() for k, v in transition.items()}
                for transition in transitions
            ]

            stamp = time.time()
            with open(str(self.home).format(stamp), 'w') as fi:
                fi.write(json.dumps(transitions))

@ray.remote
def collect(
    agent,
    env
):
    agent = energypy.make_agent(agent)
    env = energypy.make_env(env)

    env = env()
    agent = agent(env)

    obs = env.reset()
    done = False
    transitions = []
    while not done:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)

        transitions.append(
            {
                'observation': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'done': done
            }
        )
        obs = next_obs
    print(len(transitions))
    print('done')
    return transitions


def test_collect():
    agent = 'random'
    env = 'mountaincar'
    transitions = collect(agent, env)
    #  check sequense of obs -> next obs
    assert transitions[-1]['done'] == [False]


@ray.remote
class Learner():
    def learn(self, dataset):
        print('learning with {}'.format(dataset))

        #  get all objects
        dataset = [ray.get(d) for d in dataset]

        return {
            'loss': 1.0
        }


if __name__ == '__main__':
    #test_collect()

    import time
    t0 = time.time()

    ray.shutdown()
    ray.init()

    # cli
    agent = 'random'
    env = 'mountaincar'
    generations = 2
    collectors = 2

    learner = Learner.remote()

    class Dataset:
        def add

    dataset = Dataset()
    database = File('test/test_{}.json')

    for generation in range(generations):

        #  run data collection and add object ids to dataset
        #  we don't return the objects, only ids
        data = [
            collect.remote(agent, env)
            for _ in range(collectors)
        ]
        dataset.extend(data)

        #  run training & return result object
        train = learner.learn.remote(dataset)
        print(ray.get(train))

        #  save dataset locally
        #  get new data objects
        #  list of dicts
        data = [ray.get(d) for d in data]
        database.save(data)

    print('took {} seconds'.format(time.time() - t0))
