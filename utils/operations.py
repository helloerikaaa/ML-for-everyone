import math
import numpy as np


def mean_squared_error(y_true, y_pred):
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += math.pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)
