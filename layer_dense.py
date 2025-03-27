import numpy as np
import random

class Layer_Dense:
    def __init__(self, sample_size: int, neurones_number: int):
        self.weights = 0.10 * np.random.randn(sample_size, neurones_number)
        # randn is basically a gaussian distribution bounded around 0
        self.biases = np.zeros((1, neurones_number))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    