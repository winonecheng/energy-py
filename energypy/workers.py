import ray

import energypy


@ray.remote
def collect_worker(
    n_rounds,
    transition_server,
    parameter_server,
    *args,
    **kwargs
):
    """
    Runs in parallel with other collect workers
    """
    for _ in range(n_rounds):
        transition_server.add_object_id.remote(
            collect(parameter_server, *args, **kwargs)
        )


def collect(
    parameter_server,
    agent,
    env
):
    """
    Runs in sequence (in a single collect_worker)

    Gets latest parameters each time
    """
    agent = energypy.make(agent, parameter_server, env)
    env = energypy.make(env)

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

    return transitions

@ray.remote
def learn_worker(
    n_rounds,
    transition_server,
    parameter_server,
    *args,
    **kwargs
):
    for _ in range(n_rounds):
        learn(
            transition_server,
            parameter_server,
            *args,
            **kwargs
        )

def learn(
    transition_server,
    parameter_server,
    learner
):
    learner = energypy.make('add-learner', 1)

    transitions = transition_server.get_transitions.remote()
    params = parameter_server.get_params.remote()
    params = learner.learn(params, transitions)
    parameter_server.update_params.remote(params)
