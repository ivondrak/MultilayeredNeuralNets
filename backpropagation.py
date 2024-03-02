import numpy as np


class BackPropagation:
    def __init__(self, training_set, topology):
        self.training_data = training_set
        self.weights = []
        for i in range(len(topology)-1):
            self.weights.append(np.random.randn(topology[i + 1], topology[i]))
        self.learning_rate = 0.3
        self.epochs = 1000
