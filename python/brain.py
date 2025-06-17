import _binarybrain as _bb

class Brain:
    def __init__(self, n):
        self._net = _bb.Network(n)

    def tick(self, dt=0.010):
        """Advance simulation by dt seconds."""
        self._net.step(dt)

    @property
    def reward(self):
        return self._net.reward
