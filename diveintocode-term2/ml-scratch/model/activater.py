import numpy as np

class Sigmoid():
        def __init__(self):
                self.A = None
                
        def forward(self, A):
                self.A = np.copy(A)
                sig = 1/(1 - np.exp(-A))
                return sig
            
        def backward(self, dZ):
                return dZ * (1 - self.forward(np.mean(self.A, axis=0))) * self.forward(np.mean(self.A, axis=0))

class TanH():
        def __init__(self):
                self.A = None
                
        def forward(self, A):
                self.A = np.copy(A)
                #(np.exp(A) - np.exp(-A))/(np.exp(A) + np.exp(-A))
                #式変形してこっち
                return  1 - 2 / (1+ np.exp(2 * A))
            
        def backward(self, dZ):
                return dZ * (1 - np.power(self.forward(np.mean(self.A, axis=0).reshape(1, -1)), 2))

class ReLU():
        def __init__(self):
                self.A = None
                
        def forward(self, A):
                self.A = np.copy(A)
                return np.where(A >= 0, A, 0)
        
        def backward(self, dZ):
                return dZ * np.where(np.mean(self.A, axis=0).reshape(1, -1) >= 0, 1, 0)

class Softmax():
        def __init__(self):
                self.A = None
        def forward(self, A):
                self.A = np.copy(A)
                e_x = np.exp(A - np.max(A, axis=1, keepdims=True))
                return e_x/np.sum(e_x, axis=1, keepdims=True)
            
        def backward(self, Z, Y):
                return np.mean(Z - Y, axis=0).reshape(1, -1)

        def cross_entropy(self, Z, Y):
                return -(Y * (np.log(Z))).sum()/ Y.shape[0]