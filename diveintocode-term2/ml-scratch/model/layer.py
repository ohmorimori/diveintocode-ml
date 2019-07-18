import numpy as np
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