import numpy as np

class Optimizer_SGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights, biases, dweights, dbiases):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in weights]
            self.velocity_b = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * dweights[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * dbiases[i]

            weights[i] += self.velocity_w[i]
            biases[i] += self.velocity_b[i]