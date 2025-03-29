import numpy as np
from activation.relu import Activation_ReLU
from activation.softmax import Activation_SoftMax
from loss.categorical_cross_entropy import Loss_CategoricalCrossentropy
from optimizer.sgb import Optimizer_SGD
import pickle
import matplotlib.pyplot as plt

np.random.seed(0)

class NeuralNetwork():
    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int, optimizer=None):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.activations = []
        self.optimizer = optimizer if optimizer else Optimizer_SGD(learning_rate=0.01, momentum=0.9)

       # Initialize layers
        self.weights.append(0.1 * np.random.randn(self.input_size, self.hidden_layers[0]))
        self.biases.append(np.zeros((1, self.hidden_layers[0])))
        self.activations.append(Activation_ReLU())

        for i in range(len(self.hidden_layers) - 1):
            self.weights.append(0.1 * np.random.randn(self.hidden_layers[i], self.hidden_layers[i + 1]))
            self.biases.append(np.zeros((1, self.hidden_layers[i + 1])))
            self.activations.append(Activation_ReLU())  

        self.weights.append(0.1 * np.random.randn(self.hidden_layers[-1], self.output_size))
        self.biases.append(np.zeros((1, self.output_size)))
        self.activations.append(Activation_SoftMax())

    def forward(self, inputs):
        if inputs.shape[1] != self.input_size:
            raise ValueError(f"Input shape mismatch: Expected {self.input_size}, but got {inputs.shape[1]}")
        self.inputs = inputs
        self.layer_outputs = [inputs]
        for i in range(len(self.weights)):
            layer_input = self.layer_outputs[-1]
            layer_output = np.dot(layer_input, self.weights[i]) + self.biases[i]
            self.layer_outputs.append(self.activations[i].forward(layer_output))
        return self.layer_outputs[-1]

    def backward(self, dvalues):
        self.dweights = [np.zeros_like(w) for w in self.weights]
        self.dbiases = [np.zeros_like(b) for b in self.biases]
        for i in reversed(range(len(self.weights))):
            dvalues = self.activations[i].backward(dvalues)
            layer_input = self.layer_outputs[i]
            self.dweights[i] = np.dot(layer_input.T, dvalues)
            self.dbiases[i] = np.sum(dvalues, axis=0, keepdims=True)
            dvalues = np.dot(dvalues, self.weights[i].T)
        return self.dweights, self.dbiases
    
    def update_parameters(self):
        self.optimizer.update(self.weights, self.biases, self.dweights, self.dbiases)
    
    def train(self, X, y, epochs, learning_rate):
        loss_history = []
        accuracy_history = []

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        lines_loss, = ax1.plot([], [], label='Training Loss', color='blue')
        lines_accuracy, = ax2.plot([], [], label='Training Accuracy', color='red')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_xlabel('Epoch')
        ax1.set_title('Training Loss Over Epochs')
        ax2.set_title('Training Accuracy Over Epochs')
        ax1.legend()
        ax2.legend()

        for epoch in range(epochs):
            output = self.forward(X)

            loss = self.calculate_loss(y, output)
            loss_history.append(loss)  

            accuracy = self.calculate_accuracy(y, output)
            accuracy_history.append(accuracy) 

            dvalues = Loss_CategoricalCrossentropy().backward(output, y)
            self.backward(dvalues)

            self.update_parameters()

            lines_loss.set_xdata(range(len(loss_history)))
            lines_loss.set_ydata(loss_history)

            lines_accuracy.set_xdata(range(len(accuracy_history)))
            lines_accuracy.set_ydata(accuracy_history)

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()

            plt.draw()
            plt.pause(0.1)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy:.2f}%")

        plt.ioff()
        plt.show()

    def predict(self, inputs):
        output = self.forward(inputs)
        return np.argmax(output, axis=1)
    
    def calculate_loss(self, y_true, y_pred):
        return Loss_CategoricalCrossentropy().calculate(y_pred, y_true) 
    
    def calculate_accuracy(self, y_true, y_pred):
        predictions = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        return np.mean(predictions == y_true) * 100
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load_model(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)