import numpy as np


class Adagrad():
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.G = None 
        self.eps = 1e-8

    def update(self, w, grad_wrt_w):
        if self.G is None:
            self.G = np.zeros(np.shape(w))
        self.G += np.power(grad_wrt_w, 2)
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.G + self.eps)
