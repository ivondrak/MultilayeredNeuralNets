import numpy as np
class Perceptron:
    def __init__(self, num_features, num_results):
        self.training_data = []
        self.weights = []
        self.bias = 0.0
        self.learning_rate = 0.3
        self.epochs =1000

    def run(self, input):
        print("Perceptron running!")



