import numpy as np
from regression import Regression
from utils.manipulations import polynomial_features


class PolynomialRegression(Regression):
    def __init__(self, degree, iterations=1000, learning_rate=0.001):
        self.degree = degree
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(PolynomialRegression, self).__init__(
            iterations=iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        X = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).fit(X, y)

    def predict(self, X):
        X = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X)
