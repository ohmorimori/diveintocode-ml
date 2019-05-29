import numpy as np
class StochasticGradientDescent():
        """
        確率的勾配降下法
        Parameters
        ----------
        lr : 学習率
        """
        def __init__(self, lr):
                self.lr = lr
        def update(self, layer):
                """
                ある層の重みやバイアスの更新
                Parameters
                ----------
                layer : 更新前の層のインスタンス

                Returns
                ----------
                layer : 更新後の層のインスタンス
                """
                layer.W = layer.W - self.lr * layer.dW
                layer.B = layer.B - self.lr * layer.dB
                return layer
            
class AdaGrad():
        def __init__(self, lr):
                self.lr = lr
        def update(self, layer):
                
                layer.H = layer.H + np.mean(layer.dW, axis=0) ** 2

               # self.lr = self.lr * 1 / np.power(layer.H, 1/2) 
                layer.W = layer.W - self.lr * (1 / np.power(layer.H, 1/2))  * layer.dW
                layer.B = layer.B - self.lr * (1 / np.power(layer.H, 1/2))  * layer.dB
                return layer