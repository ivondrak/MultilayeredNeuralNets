import numpy as np


class BackPropagation:
    def __init__(self, training_set, topology, epochs):
        self.training_data = training_set
        self.weights = []
        for i in range(len(topology)-1):
            self.weights.append(np.random.randn(topology[i + 1], topology[i]))
        self.biases = []
        for i in range(len(topology)):
            self.biases.append(np.zeros((topology[i], 1)))
        self.deltas_biases = []
        for i in range(len(topology)):
            self.deltas_biases.append(np.zeros((topology[i], 1)))
        self.slopes = []
        for i in range(len(topology)):
            self.slopes.append(np.ones((topology[i], 1)))
        self.deltas_slopes = []
        for i in range(len(topology)):
            self.deltas_slopes.append(np.zeros((topology[i], 1)))
        self.activations = []
        for i in range(len(topology)):
            self.activations.append(np.zeros((topology[i], 1)))
        self.errors = []
        for i in range(len(topology)):
            self.errors.append(np.zeros((topology[i], 1)))
        self.deltas_weights = []
        for i in range(len(topology)-1):
            self.deltas_weights.append(np.zeros((topology[i+1], topology[i])))
        self.gradients = []
        for i in range(len(topology)-1):
            self.gradients.append(np.zeros((topology[i+1], topology[i])))
        self.num_layers = len(topology)
        self.output_error = np.zeros((topology[-1], 1))
        self.output_activation = np.zeros((topology[-1], 1))
        self.learning_rate = 0.3
        self.epochs = epochs

    def sigmoid_function(self, z, slope, bias):
        return 1 / (1 + np.exp(-1 * slope * (z - bias)))

    def sigmoid_derivative(self, y, slope):
        return slope * y * (1 - y)

    def slope_derivative(self, y, z, bias):
        return y * (1 - y) * (z - bias)

    def bias_derivative(self, y, slope):
        return -1 * slope * y * (1 - y)


    def backpropagation(self):
        for epoch in range(self.epochs):
            for data in self.training_data:
                self.feed_forward(data[0])
                self.calculate_errors(data[1])
                self.calculate_gradients()
                self.update_neural_net()

    def feed_forward(self, net_input):
        actual_input = np.array(net_input)
        self.activations[0] = actual_input.reshape(len(actual_input), 1)
        #for i in range(1, self.num_layers):
        #    self.activations[i] = self.sigmoid_function(np.dot(self.weights[i - 1], self.activations[i - 1]))
        for i in range(1, self.num_layers):
            w = self.weights[i - 1]
            s = self.slopes[i]
            b = self.biases[i]
            x = self.activations[i - 1]
            z = np.dot(w, x)
            self.activations[i] = self.sigmoid_function(z, s, b)
        self.output_activation = np.around(self.activations[-1], decimals=2)

    def calculate_errors(self, desired_output):
        array_desired_output = np.array(desired_output)
        array_desired_output = array_desired_output.reshape(len(desired_output), 1)
        self.output_error =  self.activations[-1] - array_desired_output

    def calculate_gradients(self):
        #self.errors[-1] = self.output_error * self.sigmoid_derivative(self.activations[-1])
        # Weights gradients
        self.errors[-1] = self.output_error
        self.gradients[-1] = np.dot(self.errors[-1] * self.sigmoid_derivative(self.activations[-1], self.slopes[-1]), self.activations[-2].T)
        self.calculate_weights()
        self.calculate_biases()
        self.calculate_slopes()

    def calculate_weights(self):
        self.deltas_weights[-1] = self.learning_rate * self.gradients[-1]
        for i in range(self.num_layers - 2, 0, -1):
            self.errors[i] = np.dot(self.weights[i].T, self.errors[i + 1])
            self.gradients[i - 1] = np.dot(self.errors[i] * self.sigmoid_derivative(self.activations[i], self.slopes[i]), self.activations[i - 1].T)
            self.deltas_weights[i - 1] = self.learning_rate * self.gradients[i - 1]

    def calculate_biases(self):
        self.deltas_biases[-1] = self.learning_rate * self.errors[-1] * self.bias_derivative(self.activations[-1],
                                                                                             self.slopes[-1])
        for i in range(self.num_layers - 2, 0, -1):
            self.deltas_biases[i] = self.learning_rate * self.errors[i] * self.bias_derivative(
                self.activations[i], self.slopes[i])

    def calculate_slopes(self):
        w = self.weights[-1]
        x = self.activations[-2]
        z = np.dot(w, x)
        self.deltas_slopes[-1] = self.learning_rate * self.errors[-1] * self.slope_derivative(self.activations[-1], z,
                                                                                             self.slopes[-1])
        for i in range(self.num_layers - 2, 0, -1):
            w = self.weights[i - 1]
            x = self.activations[i - 1]
            z = np.dot(w, x)
            self.deltas_slopes[i] = self.learning_rate * self.errors[i] * self.slope_derivative(
                self.activations[i], z, self.biases[i])

    def update_neural_net(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.deltas_weights[i]
        for i in range(1, len(self.biases)):
            self.biases[i] -= self.deltas_biases[i]
        for i in range(1, len(self.slopes)):
            self.slopes[i] -= self.deltas_slopes[i]

    def run(self, input_data):
        self.feed_forward(input_data)
        return self.output_activation.flatten().tolist()