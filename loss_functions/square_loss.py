import numpy as np
from loss_functions.loss import Loss


class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def grad(self, y, y_pred):
        return -(y - y_pred)
