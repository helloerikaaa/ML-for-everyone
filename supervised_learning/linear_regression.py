import numpy as np
from regression import Regression


class LinearRegression(Regression):
    def __init__(self, iterations=1000, learning_rate=0.001, gradient_descent=True):
        self.grad = gradient_descent
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(
            iterations=iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        if not self.grad:
            X = np.insert(X, 0, 1, axis=1)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)
