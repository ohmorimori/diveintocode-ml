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
