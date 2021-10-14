import numpy as np


class Sigmoid():
    def __call__(self, x):
        return 1 / (np.exp(-x))

    def grad(self, x):
        return self.__call__(x) * (1 - self.__call__(x))
