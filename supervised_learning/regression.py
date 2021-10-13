import math
import numpy as np


class Regression(object):
    def __init__(self, iterations, learning_rate):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, features):
        limit = 1 / math.sqrt(features)
        self.w = np.random.uniform(-limit, limit, (features, ))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.training_erros = []
        self.initialize_weights(features=X.shape[1])

        for _ in range(self.iterations):
            y_pred = X.dot(self.w)
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_erros.append(mse)

            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred
