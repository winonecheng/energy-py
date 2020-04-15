import ray


class AddLearner():
    def __init__(self, step):
        self.step = step

    def learn(self, parameters, transitions):
        return parameters + self.step


