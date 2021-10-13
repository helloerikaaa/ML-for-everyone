import numpy as np


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError

    def grad(self, y, y_pred):
        return NotImplementedError

    def acc(self, y, y_pred):
        return NotImplementedError


class SquareLoss(Loss):
    def __init__(self): pass
    
    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)
    
    def grad(self, y, y_pred):
        return -(y - y_pred)
