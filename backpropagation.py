import numpy as np


class BackPropagation:
    def __init__(self, training_set, topology):
        self.training_data = training_set
        self.weights = []
        for i in range(len(topology)-1):
            self.weights.append(np.random.randn(topology[i + 1], topology[i]))
        #self.biases = []
        #for i in range(1, len(topology)):
        #    self.biases.append(np.random.randn(1, topology[i]))
        #self.slopes = []
        #for i in range(1, len(topology)):
        #    self.biases.append(np.random.randn(1, topology[i]))
        self.activations = []
        for i in range(len(topology)):
            self.activations.append(np.zeros((topology[i], 1)))
        self.errors = []
        for i in range(len(topology)-1):
            self.errors.append(np.zeros((topology[i+1], 1)))
        self.deltas = []
        for i in range(len(topology)-1):
            self.deltas.append(np.zeros((topology[i+1], 1)))
        self.gradients = []
        for i in range(len(topology)-1):
            self.gradients.append(np.zeros((topology[i+1], topology[i])))
        self.num_layers = len(topology)
        self.output_error = np.zeros((1, topology[-1]))
        self.output_activation = np.zeros((1, topology[-1]))
        self.learning_rate = 0.3
        self.epochs = 1000

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):


    def backpropagation(self):
        for epoch in range(self.epochs):
            for data in self.training_data:
                self.feed_forward(data[0])
                self.calculate_errors(data[1])
                self.calculate_gradients(data)
                self.update_weights()

    def feed_forward(self, net_input):
        actual_input = np.array(net_input)
        self.activations[0] = actual_input.reshape(len(actual_input), 1)
        for i in range(1, self.num_layers):
            self.activations[i] = self.sigmoid_function(np.dot(self.weights[i - 1], self.activations[i - 1]))

    def calculate_errors(self, desired_output):
        array_desired_output = np.array(desired_output)
        array_desired_output = array_desired_output.reshape(len(desired_output), 1)
        self.output_error =  self.activations[-1] - array_desired_output

    def calculate_gradients(self):
        for i in range(self.num_layers - 2, -1, -1):
            self.errors[i] = np.dot(self.weights[i + 1].T, self.errors[i + 1]) * self.sigmoid_derivative(
                self.activations[i + 1])
            self.gradients[i] = np.dot(self.errors[i], self.activations[i].T)
            self.deltas[i] = self.learning_rate * self.gradients[i]
            self.weights[i] -= self.deltas[i]

    def update_weights(self):









