
from collectors import *

import timeit

def test_collectors():
    start = timeit.default_timer()
    num = 8

    env_id = 'cartpole-v0'
    policy_params = {
        'policy_id': 'random',
        'env_id': env_id,
        'weights_dir': 'a'
    }

    collectors = [
        SingleProcessCollector(
            policy_params, env_id
        ),
        MultiProcessCollector(
            policy_params, env_id, n_jobs=2
        )
    ]

    for collector in collectors:
        data = collector.collect(num)
        assert len(data) == num
        stop = timeit.default_timer()
        print(stop - start)

        rews = [np.sum(d['reward']) for d in data]
        assert np.mean(rews) != rews[0]
