

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError

    def grad(self, y, y_pred):
        return NotImplementedError

    def acc(self, y, y_pred):
        return NotImplementedError
