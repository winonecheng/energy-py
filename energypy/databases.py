import os


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
