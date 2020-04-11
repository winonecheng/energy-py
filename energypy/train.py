import json
import os
import pathlib
import time

import numpy as np
import ray

import energypy


def main(agent, env, learner, n_generations, n_collectors):
    print('training starting')

    t0 = time.time()

    ray.shutdown()
    ray.init()

    learner = energypy.make(learner)

    database = File('test/test_{}.json')

    dataset = []
    parameters = {}

    for generation in range(n_generations):

        #  run data collection and add object ids to dataset
        #  we don't return the objects, only ids
        data = [
            collect.remote(agent, parameters, env)
            for _ in range(n_collectors)
        ]

        #  run training & return result object
        parameters = learner.learn.remote(parameters, dataset)

        #  extend after so that learning can happen in parallel with collect
        dataset.extend(data)

        #  save dataset locally
        #  get new data objects
        #  list of dicts
        data = [ray.get(d) for d in data]
        database.save(data)

        #  save agent

    print('training took {} seconds'.format(time.time() - t0))


if __name__ == '__main__':
    main('random', 'mountaincar', 10, 2)
