import numpy as np
from layer_dense import Layer_Dense

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, 1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

layer1 = Layer_Dense(sample_size=4, neurones_number=5)
layer2 = Layer_Dense(sample_size=5, neurones_number=2)

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
# print(layer2.output)
