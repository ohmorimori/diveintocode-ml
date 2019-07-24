
import numpy as np
class Relu():
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = np.copy(x)
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Softmax():
    def __init__(self):
        pass

    def forward(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        #softmax
        out = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return out

    def backward(self, dout):

        batch_size = dout.shape[0]
        dx = (dout) / batch_size
        return dx
