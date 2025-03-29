from neural_network import NeuralNetwork
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

nn = NeuralNetwork(input_size=28*28, hidden_layers=[128, 64], output_size=10)
nn.train(X_train, y_train, epochs=100, learning_rate=0.01)

nn.save_model('trained_model.pkl')

accuracy = nn.calculate_accuracy(y_test, nn.forward(X_test))
print(f"Test Accuracy: {accuracy:.2f}%")