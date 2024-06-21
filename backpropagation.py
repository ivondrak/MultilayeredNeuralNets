import numpy as np

# Weights only adaptation
class GenericBackPropagation:

    def __init__(self, training_set, topology, learning_rates, epochs):
        self.training_data = training_set
        self.weights = []
        for i in range(len(topology)-1):
            self.weights.append(np.random.rand(topology[i + 1], topology[i]))
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
        self.learning_rates = learning_rates
        self.epochs = epochs
        self.history = []

    def activation_function(self, z):
        return 1 / (1 + np.exp(-1 * z))

    def activation_derivative(self, y):
        return y * (1 - y)

    def backpropagation(self):
        self.history = []
        for epoch in range(self.epochs):
            for data in self.training_data:
                self.feed_forward(data[0])
                self.calculate_errors(data[1])
                self.calculate_gradients()
                self.update_neural_net()
            mean_squared_error = self.calculate_mean_squared_error()
            self.history.append(mean_squared_error)

    def feed_forward(self, net_input):
        actual_input = np.array(net_input)
        self.activations[0] = actual_input.reshape(len(actual_input), 1)
        for i in range(1, self.num_layers):
            w = self.weights[i - 1]
            x = self.activations[i - 1]
            z = w @ x
            self.activations[i] = self.activation_function(z)
        self.output_activation = np.around(self.activations[-1], decimals=2)

    def calculate_errors(self, desired_output):
        array_desired_output = np.array(desired_output)
        array_desired_output = array_desired_output.reshape(len(desired_output), 1)
        self.output_error =  self.activations[-1] - array_desired_output

    def calculate_gradients(self):
        self.errors[-1] = self.output_error
        self.gradients[-1] = (self.errors[-1] * self.activation_derivative(self.activations[-1])) @ self.activations[-2].T
        self.calculate_weights()

    def calculate_weights(self):
        self.deltas_weights[-1] = self.learning_rates[0] * self.gradients[-1]
        for i in range(self.num_layers - 2, 0, -1):
            self.errors[i] = self.weights[i].T @ self.errors[i + 1]
            self.gradients[i - 1] = (self.errors[i] * self.activation_derivative(self.activations[i])) @ self.activations[i - 1].T
            self.deltas_weights[i - 1] = self.learning_rates[0] * self.gradients[i - 1]

    def update_neural_net(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.deltas_weights[i]

    def run(self, input_data):
        self.feed_forward(input_data)
        return self.output_activation.flatten().tolist()
    
    def calculate_mean_squared_error(self):
        total_error = 0
        for data in self.training_data:
            input_data = data[0]
            desired_output = data[1]
            actual_output = self.run(input_data)
            error = np.sum((np.array(actual_output) - np.array(desired_output)) ** 2)
            total_error += error
        mean_squared_error = total_error / len(self.training_data)
        return mean_squared_error

    def calculate_max_error(self):
        max_error = 0
        for data in self.training_data:
            input_data = data[0]
            desired_output = data[1]
            predicted_output = self.run(input_data)
            error = np.max(np.abs(np.array(desired_output) - np.array(predicted_output)))
            max_error = max(max_error, error)
        return max_error  

# Weights, biases, and slopes adaptation
class BackPropagation (GenericBackPropagation):
    def __init__(self, training_set, topology, learning_rates, epochs):
        self.training_data = training_set
        self.weights = []
        for i in range(len(topology)-1):
            self.weights.append(np.random.rand(topology[i + 1], topology[i]))
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
        self.learning_rates = learning_rates
        self.epochs = epochs
        self.history = []

    def activation_function(self, z, slope, bias):
        return 1 / (1 + np.exp(-1 * slope * (z - bias)))

    def activation_derivative(self, y, slope):
        return slope * y * (1 - y)

    def slope_derivative(self, y, z, bias):
        return y * (1 - y) * (z - bias)

    def bias_derivative(self, y, slope):
        return -1 * slope * y * (1 - y)

    def feed_forward(self, net_input):
        actual_input = np.array(net_input)
        self.activations[0] = actual_input.reshape(len(actual_input), 1)
        for i in range(1, self.num_layers):
            w = self.weights[i - 1]
            s = self.slopes[i]
            b = self.biases[i]
            x = self.activations[i - 1]
            z = w @ x
            self.activations[i] = self.activation_function(z, s, b)
        self.output_activation = np.around(self.activations[-1], decimals=2)

    def calculate_gradients(self):
        self.errors[-1] = self.output_error
        self.gradients[-1] = (self.errors[-1] * self.activation_derivative(self.activations[-1], self.slopes[-1])) @ self.activations[-2].T
        self.calculate_weights()
        self.calculate_biases()
        self.calculate_slopes()

    def calculate_weights(self):
        self.deltas_weights[-1] = self.learning_rates[0] * self.gradients[-1]
        for i in range(self.num_layers - 2, 0, -1):
            self.errors[i] = self.weights[i].T @ self.errors[i + 1]
            # self.gradients[i - 1] = np.dot(self.errors[i] * self.sigmoid_derivative(self.activations[i], self.slopes[i]), self.activations[i - 1].T)
            self.gradients[i - 1] = (self.errors[i] * self.activation_derivative(self.activations[i], self.slopes[i])) @ self.activations[i - 1].T
            self.deltas_weights[i - 1] = self.learning_rates[0] * self.gradients[i - 1]

    def calculate_biases(self):
        self.deltas_biases[-1] = self.learning_rates[1] * self.errors[-1] * self.bias_derivative(self.activations[-1],
            self.slopes[-1])
        for i in range(self.num_layers - 2, 0, -1):
            self.deltas_biases[i] = self.learning_rates[1] * self.errors[i] * self.bias_derivative(
                self.activations[i], self.slopes[i])

    def calculate_slopes(self):
        w = self.weights[-1]
        x = self.activations[-2]
        z = np.dot(w, x)
        self.deltas_slopes[-1] = self.learning_rates[2] * self.errors[-1] * self.slope_derivative(self.activations[-1], z,
            self.slopes[-1])
        for i in range(self.num_layers - 2, 0, -1):
            w = self.weights[i - 1]
            x = self.activations[i - 1]
            z = w @ x
            self.deltas_slopes[i] = self.learning_rates[2] * self.errors[i] * self.slope_derivative(
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
    
    def calculate_mean_squared_error(self):
        total_error = 0
        for data in self.training_data:
            input_data = data[0]
            desired_output = data[1]
            actual_output = self.run(input_data)
            error = np.sum((np.array(actual_output) - np.array(desired_output)) ** 2)
            total_error += error
        mean_squared_error = total_error / len(self.training_data)
        return mean_squared_error

    def calculate_max_error(self):
        max_error = 0
        for data in self.training_data:
            input_data = data[0]
            desired_output = data[1]
            predicted_output = self.run(input_data)
            error = np.max(np.abs(np.array(desired_output) - np.array(predicted_output)))
            max_error = max(max_error, error)
        return max_error  

# ReLU activation function   
 
class ReLUBackPropagation(GenericBackPropagation):
    
    def __init__(self, training_set, topology, learning_rates, epochs):
        self.training_data = training_set
        self.weights = []
        # He initialization
        for i in range(len(topology)-1):
            self.weights.append(np.random.randn(topology[i + 1], topology[i]) * np.sqrt(2.0/topology[i]))
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
        self.learning_rates = learning_rates
        self.epochs = epochs
        self.history = []

    def activation_function(self, z):
        return np.maximum(0, z)

    def activation_derivative(self, y):
        return np.where(y > 0, 1, 0)
    
# Softmax activation function
class SoftmaxBackPropagation(ReLUBackPropagation):

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum(axis=0)

    def cross_entropy_loss(self, y_pred, y_true):

        total_loss = 0
        for data in self.training_data:
            input_data = data[0]
            desired_output = data[1]
            actual_output = self.run(input_data)
            loss = -np.sum(np.array(desired_output) * np.log(np.array(actual_output) + 1e-8))
            total_loss += loss
        #mean_squared_error = total_error / len(self.training_data)
        return total_loss
    
    def feed_forward(self, net_input):
        actual_input = np.array(net_input)
        self.activations[0] = actual_input.reshape(len(actual_input), 1)
        for i in range(1, self.num_layers - 1):
            w = self.weights[i - 1]
            x = self.activations[i - 1]
            z = w @ x
            self.activations[i] = self.activation_function(z)
        # Softmax on the last layer
        w = self.weights[-1]
        x = self.activations[-2]
        z = w @ x
        self.activations[-1] = self.softmax(z)
        self.output_activation = np.around(self.activations[-1], decimals=2)

    def backpropagation(self):
        self.history = []
        for epoch in range(self.epochs):
            for data in self.training_data:
                self.feed_forward(data[0])
                self.calculate_errors(data[1])
                self.calculate_gradients()
                self.update_neural_net()
            cross_entropy_loss = self.cross_entropy_loss(self.output_activation, data[1])
            self.history.append(cross_entropy_loss)

    def calculate_gradients(self):
        self.errors[-1] = self.output_error
        self.gradients[-1] = self.errors[-1]
        self.calculate_weights()

    def calculate_weights(self):
        self.deltas_weights[-1] = self.learning_rates[0] * self.gradients[-1]
        for i in range(self.num_layers - 2, 0, -1):
            self.errors[i] = self.weights[i].T @ self.errors[i + 1]
            self.gradients[i - 1] = (self.errors[i] * self.activation_derivative(self.activations[i])) @ self.activations[i - 1].T
            self.deltas_weights[i - 1] = self.learning_rates[0] * self.gradients[i - 1]
    

