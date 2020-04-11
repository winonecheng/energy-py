import time
import ray

from energypy.workers import collect_worker, learn_worker
from energypy.servers import ParameterServer, TransitionServer

ray.init(local_mode=True)
ts = TransitionServer.remote()
ps = ParameterServer.remote(0)

n_episodes = 40
n_actors = 2
n_actor_rounds = 10
n_generations = int(n_episodes / (n_actors * n_actor_rounds))

n_learners = 2
n_learning_rounds = 2

t = []
p = 0
t0 = time.time()
for g in range(n_generations):
    actors = [
        collect_worker.remote(n_actor_rounds, ts, ps, 'random', 'noenv')
        for _ in range(n_actors)
    ]

    learners = [
        learn_worker.remote(n_learning_rounds, ts, ps, 'add-learner')
        for _ in range(n_learners)
    ]

while len(t) < n_generations * n_actors * n_actor_rounds:
    objs = ts.get_object_ids.remote()
    t = ray.get(objs)
    print(len(t))
    print(time.time() - t0)

#ps.update_params.remote(10)
params = ps.get_params.remote()
p = ray.get(params)
print(p)
