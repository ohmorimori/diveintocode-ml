import numpy as np
class SimpleInitializer:
        """
        ガウス分布によるシンプルな初期化
        Parameters
        ----------
        sigma : float
          ガウス分布の標準偏差
        """
        def __init__(self, sigma):
                self.sigma = sigma
                
        def coef(self, n_nodes1, n_nodes2):
                """
                重みの初期化
                Parameters
                ----------
                n_nodes1 : int
                  前の層のノード数
                n_nodes2 : int
                  後の層のノード数

                Returns
                ----------
                W :
                """
                
                W = self.sigma * np.random.randn(n_nodes1, n_nodes2)
                B = self.sigma * np.random.randn(1, n_nodes2)
                H = np.mean(W, axis=0) ** 2
                return W, B, H


class XavierInitializer():
        def __init__(self):
                self.sigma = None
        def coef(self, n_nodes1,  n_nodes2):
                self.sigma = 1/np.power(n_nodes1, 1/2)
                W = np.random.normal(loc=0.0, scale=self.sigma, size=(n_nodes1,  n_nodes2))
                B = np.random.normal(loc=0.0, scale=self.sigma, size=(1,  n_nodes2))
                H = np.mean(W, axis=0) ** 2
                return W, B, H
            
class HeInitializer():
        def __init__(self):
                self.sigma = None
        
        def coef(self, n_nodes1, n_nodes2):
                self.sigma = np.power(2 / n_nodes1, 1/2)
                W = np.random.normal(loc=0.0, scale=self.sigma, size=(n_nodes1, n_nodes2))
                B = np.random.normal(loc=0.0, scale=self.sigma, size=(1, n_nodes2))
                H = np.mean(W, axis=0) ** 2
                return W, B, H