import numpy as np

class TransportConnection:
    def __init__(self, from_node, to_node, mode, duration, frequency):
        self.from_node = from_node
        self.to_node = to_node
        self.mode = mode
        self.duration = duration  # in minutes
        self.frequency = frequency  # in minutes
        self.reliability = np.random.uniform(0.8, 1.0)  # historical reliability score