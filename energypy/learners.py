import ray


class AddLearner():
    def __init__(self, step):
        self.step = step

    def learn(self, parameters, transitions):
        parameters = ray.get(parameters)
        return parameters + self.step


