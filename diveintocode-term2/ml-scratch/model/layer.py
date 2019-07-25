import sys
dir_str = "../../ml-scratch/utils"
if (dir_str not in sys.path):
    sys.path.append(dir_str)

import numpy as np
from change_shape import im2col, col2im
class FullyConnectedLayer:
    """
    ノード数n_nodes1からn_nodes2への全結合層
    Parameters
    ----------
    n_nodes1 : int
      前の層のノード数
    n_nodes2 : int
      後の層のノード数
    initializer : 初期化方法のインスタンス
    optimizer : 最適化手法のインスタンス
    """
    def __init__(self, n_nodes1, n_nodes2, initializer, optimizer):
        self.optimizer = optimizer
        # 初期化
        # initializerのメソッドを使い、self.Wとself.Bを初期化する
        self.W, self.B, self.H  = initializer.coef(n_nodes1, n_nodes2)
        self.pre_Z = None

        self.dA = None
        self.dB = None
        self.dW = None
        self.d_pre_Z = None

    def forward(self, X):
        """
        フォワード
        Parameters
        ----------
        X : 次の形のndarray, shape (batch_size, n_nodes1)
            入力
        Returns
        ----------
        A : 次の形のndarray, shape (batch_size, n_nodes2)
            出力
        """
        self.pre_Z = np.copy(X)
        A = self.pre_Z @ self.W + self.B
        return A

    def backward(self, dA):
        """
        バックワード
        Parameters
        ----------
        dA : 次の形のndarray, shape (batch_size, n_nodes2)
            後ろから流れてきた勾配
        Returns
        ----------
        dZ : 次の形のndarray, shape (batch_size, n_nodes1)
            前に流す勾配
        """
        self.dB = dA
        self.d_pre_Z = (dA @ self.W.T)
        self.dW = np.mean(self.pre_Z.T, axis=1).reshape(-1, 1) @  dA

        # 更新
        self = self.optimizer.update(self)

        return self.d_pre_Z

class ScratchRNN():
    """
    ノード数n_nodes1からn_nodes2への全結合層
    Parameters
    ----------
    n_nodes : int
      ノード数
    initializer : 初期化方法のインスタンス
    optimizer : 最適化手法のインスタンス
    """
    def __init__(self, n_nodes, optimizer, initializer, activator, lr):
        self.optimizer = optimizer
        self.initializer = initializer
        self.activator = activator

        self.Wx = None #weight for input at time t. shape(n_features, n_nodes)
        self.Wh = None #weight for condition at time t-1. shape(n_nodes, n_nodes)
        self.B = None #bias
        self.n_nodes = n_nodes
        self.t = 0 #time (sequence)

        self.A = None #condition before activation. shape(batch_size, n_nodes)
        self.lr = lr
        #slope
        self.dA = None
        self.dB = None
        self.dWx = None
        self.dWh = None
        self.d_pre_H = None

    def initialize(self):
        #test
        #self.Wx = np.array([[1, 3, 5, 7], [3, 5, 7, 8]])/100
        #self.Wh = np.array([[1, 3, 5, 7], [2, 4, 6, 8], [3, 5, 7, 8], [4, 6, 8, 10]])/100
        #self.B = np.array([1])
        self.Wx, self.B, _ = self.initializer.coef(n_nodes1=self.n_features, n_nodes2=self.n_nodes)
        self.Wh, _, _ = self.initializer.coef(n_nodes1=self.n_nodes, n_nodes2=self.n_nodes)
        self.H = np.zeros((self.batch_size, self.n_sequences, self.n_nodes))
        self.A = np.zeros((self.batch_size, self.n_sequences, self.n_nodes))
        self.dA = np.zeros_like(self.A)
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.dB = np.zeros_like(self.B)
        self.pre_dH = np.zeros_like(self.H)
        self.pre_dX = np.zeros_like(self.A)

    def forward(self, X):
        """
        forward propagation

        Parameters
        ---------------
        X: ndarray of shape (batch_size, n_sequences, n_features)
            input
        pre_X: ndarray of shape(batch_size, n_nodes)
            activated output of pre layer
        Returns
        ---------------
        A: ndarray of shape (batch_size, n_nodes)
            output
        """
        self.X = np.copy(X)
        #tが0の時
        if (self.t == 0):
            self.batch_size, self.n_sequences, self.n_features = self.X.shape
            #initializeして最初のAとZを計算
            self.initialize()
            self.A[:, 0, :] = self.X[:, 0, :] @ self.Wx + np.zeros((self.batch_size, self.n_nodes)) @ self.Wh + self.B
            self.H[:, 0, :] = self.activator.forward(self.A[:, 0, :])

        #時刻を進める
        self.t += 1
        self.X = np.copy(X)
        #tがlimitに達したら再帰終了
        if (self.t == self.n_sequences -1):
            self.t = 0
            return self.H

        #tがlimit未満なら以下を実行
        self.A[:, self.t, :] = X[:, self.t, :] @ self.Wx + self.H[:, self.t - 1, :] @ self.Wh + self.B
        #activate
        self.H[:, self.t, :] = self.activator.forward(self.A[:, self.t, :])

        return self.forward(self.X)

    def backward(self, dZ):
        """
        backward propagation
        Parameters
        ----------
        dZ : ndarray of shape (batch_size, n_nodes1)
            後ろから流れてきた勾配
        Returns
        ----------
        pre_dH : 次の形のndarray, shape (batch_size, n_nodes1)
            前に流す勾配
        """

        #activate
        self.dA[:, self.t, :] = self.activator.backward(dZ)
        #tがlimitに達したら再帰終了
        if (self.t == self.n_sequences -1):
            self.t = 0
            return self.pre_dH

        #update coef
        self.dB = np.mean(self.dA[:, self.t, :], axis=0).reshape(1, -1)
        self.dWx = np.mean(self.X[:, self.t, :].T, axis=1).reshape(-1, 1) @  self.dA[:, self.t, :]

        self.pre_dX[:, self.t, :] = np.mean(self.dA[:, self.t, :] @ self.Wx.T, axis=1)
        if (self.t > 0):
            self.dWh = np.mean(self.H[:, self.t - 1, :].T, axis=1).reshape(-1, 1) @  self.dA[:, self.t, :]
            self.pre_dH[:, self.t-1, :] = (self.dA[:, self.t, :] @ self.Wh.T)

        # 更新
        self._update_weights()
        #時刻を進める
        self.t += 1
        #再帰
        return self.backward(dZ)

    def _update_weights(self):
        self.Wx -=  self.lr * self.dWx
        self.Wh -= self.lr * self.dWh
        self.B -= self.lr * self.dB

class Conv2D():
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x =None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        #filter shape
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)
        col = im2col(input_data=x, filter_h=FH, filter_w=FW, stride=self.stride, pad=self.pad)
        col_W = self.W.reshape(FN, -1).T #フィルターを展開
        out = col @ col_W + self.b
        self.x = x
        self.col = col
        self.col_W = col_W

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1)
        dout = dout.reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = self.col.T @ dout
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = dout @ self.col_W.T
        dx = col2im(col=dcol, input_shape=self.x.shape, filter_h=FH, filter_w=FW, stride=self.stride, pad=self.pad)
        return dx

class MaxPooling2D():
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max=None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size, ))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[2], -1)
        dx = col2im(col=dcol, input_shape=self.x.shape, filter_h=self.pool_h, filter_w=self.pool_w, stride=self.stride, pad=self.pad)
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):

        self.x = x
        out = self.x @ self.W + self.b
        return out

    def backward(self, dout):
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dx

class Flatten():
    def __init__(self):
        self.prev_layer_shape = None

    def forward(self, x):
        self.prev_layer_shape = x.shape
        #(N, C, H, W)を(N, C*H*W)に
        return x.reshape((x.shape[0], -1))

    def backward(self, dout):
        #(N, C*H*W)を(N, C, H, W)に
        return (dout.reshape(self.prev_layer_shape))


