import numpy as np


class Adadelta():
    def __init__(self, rho=0.95, eps=1e-6):
        self.E_w_updt = None 
        self.E_grad = None  
        self.w_updt = None 
        self.eps = eps
        self.rho = rho

    def update(self, w, grad_wrt_w):
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))
            self.E_w_updt = np.zeros(np.shape(w))
            self.E_grad = np.zeros(np.shape(grad_wrt_w))

        self.E_grad = self.rho * self.E_grad + \
            (1 - self.rho) * np.power(grad_wrt_w, 2)

        RMS_delta_w = np.sqrt(self.E_w_updt + self.eps)
        RMS_grad = np.sqrt(self.E_grad + self.eps)

        adaptive_lr = RMS_delta_w / RMS_grad
        self.w_updt = adaptive_lr * grad_wrt_w
        self.E_w_updt = self.rho * self.E_w_updt + \
            (1 - self.rho) * np.power(self.w_updt, 2)

        return w - self.w_updt
