import numpy as np

from activations import Sigmoid, Softmax
from dense import Dense


class NeuralNetwork:

    def __init__(self):
        self.network = [
            Dense(28 * 28, 50),
            Sigmoid(),
            Dense(50, 10),
            Sigmoid()
        ]
        self.epochs = 1000
        self.learning_rate = 0.12
        self.verbose = True
        self.loss = mse
        self.loss_prime = mse_prime
        self.train_loss = []
        self.test_score = 0

    def fit(self, X, Y):
        for e in range(self.epochs):
            error = 0
            for x, y in zip(X, Y):
                #forward
                output = self.predict(x)

                #error
                error += self.loss(y, output)

                #backward
                grad = self.loss_prime(y, output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad, self.learning_rate)

            error /= len(X)
            self.train_loss.append(error)
            if e % 100 == 0:
                if self.verbose:
                    print(f"{e + 1}/{self.epochs}, error={error}")

    def predict(self, X):
        output = X
        for layer in self.network:
            output = layer.forward(output)
        return output

    def score(self, X, Y):
        matches = 0
        for x, y in zip(X, Y):
            output = self.predict(x)
            if np.argmax(output) == np.argmax(y):
                matches += 1

        self.test_score = (matches / X.shape[0] * 100)
        print((matches / X.shape[0] * 100))


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
