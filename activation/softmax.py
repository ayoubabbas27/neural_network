import numpy as np

class Activation_SoftMax():
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = norm_values
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        for i in range(len(dvalues)):
            jacobian_matrix = np.diag(self.output[i])
            jacobian_matrix -= np.dot(self.output[i].reshape(-1, 1), self.output[i].reshape(1, -1))
            self.dinputs[i] = np.dot(jacobian_matrix, dvalues[i])
        return self.dinputs