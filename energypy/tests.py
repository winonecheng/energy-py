import energypy


def test_collect():
    agent = 'random'
    env = 'mountaincar'
    transitions = energypy.workers.collect(agent, env)
    #  check sequense of obs -> next obs
    assert transitions[-1]['done'] == [False]

def test_random_agent():
    env = energypy.make('mountaincar')
    agent = energypy.make('random', None, env)
    agent.act(0)


def test_mountain_car_wrapper():
    env = energypy.make('mountaincar')
    env.seed(42)
    obs = env.reset()
    obs, r, done, info = env.step(env.action_space.sample())
