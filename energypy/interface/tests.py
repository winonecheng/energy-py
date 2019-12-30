
from collectors import *

import timeit

def test_collectors():
    start = timeit.default_timer()
    num = 1024

    collectors = [
        SingleProcessCollector(RandomPolicy('cartpole-v0')),
        MultiProcessCollector(RandomPolicy('cartpole-v0'),  n_jobs=8)
    ]

    for collector in collectors:
        data = collector.collect(num)
        assert len(data) == num
        stop = timeit.default_timer()
        print(stop - start)

        rews = [np.sum(d['reward']) for d in data]
        assert np.mean(rews) != rews[0]
