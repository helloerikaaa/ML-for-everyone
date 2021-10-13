import math
import numpy as np
from tqdm import tqdm


from utils.activation_functions import Sigmoid
from utils.loss_functions import SquareLoss

class Perceptron():

    def __init__(self, iterations=1000, activation_function=Sigmoid, loss=SquareLoss, learning_rate=0.01):
        self.iterations = iterations
        self.activation_function = activation_function()
        self.loss = loss()
        self.learning_rate = learning_rate

    def train(self, X, y):
        _, features = np.shape(X)
        _, outputs = np.shape(y)

        # Weight inicialization
        limit = 1 / math.sqrt(features)
        self.W = np.random.uniform(-limit, limit, (features, outputs))
        self.W0 = np.zeros((1, outputs))

        for _ in tqdm(range(self.iterations)):
            linear_output = X.dot(self.W) + self.W0
            y_pred = self.activation_function(linear_output)

            # Gradient of loss calculation
            error_gradient = self.loss.grad(
                y, y_pred) * self.activation_function.grad(linear_output)
            grad_wrt_w = X.T.dot(error_gradient)
            grad_wrt_w0 = np.sum(error_gradient, axis=0, keepdims=True)

            # Updating weights
            self.W -= self.learning_rate * grad_wrt_w
            self.W0 -= self.learning_rate * grad_wrt_w0

    def test(self, X):
        # test the trained model
        y_pred = self.activation_function(X.dot(self.W) + self.W0)
        return y_pred
