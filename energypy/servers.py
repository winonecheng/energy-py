import ray


@ray.remote
class TransitionServer:
    def __init__(self):
        self.object_ids = []

    def get_object_ids(self):
        return self.object_ids

    def add_object_id(self, new):
        self.object_ids.append(new)

    def get_transitions(self):
        objs = ts.get_object_ids.remote()
        return(ray.get(objs))


@ray.remote
class ParameterServer:
    def __init__(self, params):
        self.params = params

    def get_params(self):
        # print('getting params')
        return self.params

    def update_params(self, new):
        print('update params')
        self.params = new
