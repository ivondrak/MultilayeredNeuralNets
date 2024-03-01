import numpy as np


class Perceptron:
    def __init__(self, training_set):
        self.training_data = training_set
        num_features = len(training_set[0][0])
        num_results = len(training_set[0][1])
        self.weights = np.random.randn(num_results, num_features + 1)
        self.learning_rate = 0.3
        self.epochs = 1000

    def signum_function(self, x):
        return (x >= 0).astype(np.int32)

    def learning(self):
        for epoch in range(self.epochs):
            for vector in self.training_data:
                input_vector = np.array(vector[0])
                desired_output = np.array(vector[1])
                actual_input = np.insert(input_vector, 0, -1.0)
                product = np.dot(self.weights, actual_input)
                actual_output = self.signum_function(product)
                error = desired_output - actual_output
                self.weights += np.dot(self.learning_rate * error.reshape([2, 1]), actual_input.reshape([1, 4]))

    def run(self, net_input):
        actual_input = np.insert(net_input, 0, -1.0)
        product = np.dot(self.weights, actual_input)
        actual_output = self.signum_function(product)
        print("Weights are: ", self.weights)
        print("Input vector is: ", actual_input)
        print("Output vector is: ", product)
        print("Output vector with applied sign function is: ", actual_output)


if __name__ == "__main__":
    training_set = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [1])]
    perceptron = Perceptron(training_set)
    perceptron.learning()
    perceptron.run([0, 0])
    perceptron.run([0, 1])
    perceptron.run([1, 0])
    perceptron.run([1, 1])