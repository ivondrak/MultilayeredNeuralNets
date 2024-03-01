# This is a sample Python script.
from perceptron import Perceptron


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

training_set = [
    ([0.0, 1.0, 1.0], [0.0, 0.0]),
    ([1.0, 0.0, 0.0], [1.0, 1.0]),
    ([0.5, 0.5, 0.5], [1.0, 1.0])
]

net_input = [1.0, 0.0, 0.0]


def run_perceptron():

    # Use a breakpoint in the code line below to debug your script.
    perceptron = Perceptron(training_set)
    perceptron.learning()
    perceptron.run([net_input])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_perceptron()
    training_set = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [1])]
    perceptron = Perceptron(training_set)
    perceptron.learning()
    perceptron.run([0, 0])
    perceptron.run([0, 1])
    perceptron.run([1, 0])
    perceptron.run([1, 1])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
