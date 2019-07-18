import sys

model = "../../ml-scratch/model"
utils =  "../../ml-scratch/utils"
if model not in sys.path:
        sys.path.append(model)
if utils not in sys.path:
        sys.path.append(utils) 

from get_mini_batch import GetMiniBatch
from layer import FullyConnectedLayer, ScratchRNN
from initializer import HeInitializer
from optimizer import AdaGrad
from activator import TanH
from activator import Softmax

import time
import numpy as np

class ScratchRNNClassifier():
        def __init__(self, n_nodes1= 400, n_output = 10,  sigma=0.01, lr=0.001, batch_size=20, n_epochs=30):
                self.n_nodes1 = n_nodes1
                self.n_output = n_output
                self.n_sequences = None
                self.n_features = None
                self.sigma = sigma
                self.lr = lr
                self.batch_size = batch_size
                self.entropy= {"training": [], "validation": []}
                self.n_epochs = n_epochs
                
        def fit(self, X, y, X_val=None, y_val=None):
                X = np.array(X)
                y = np.array(y)
                
                _, _, self.n_features = X.shape
                
                #layerを作る
                self._gen_layers()
                #one-hot
                eye = np.eye(len(np.unique(y)))
                start = time.time()
                #epoch毎に
                for epoch in range(self.n_epochs):
                        #mini batch生成
                        mini_batch = GetMiniBatch(X, y, self.batch_size)

                        #mini batch毎に
                        train_entrpy = []
                        for mini_X, mini_y in mini_batch:
                                #forward propagation
                                self._propagate_forward(X=mini_X)
                                #entropy計算
                                mini_entrpy = self._calc_entropy(y=eye[mini_y.reshape(-1,)])
                                #backward propagation
                                self._propagate_backward(y=eye[mini_y.reshape(-1,)])
                                #mini batchごとのentropy保管
                                train_entrpy.append(mini_entrpy)
                        
                        #batchごとのentropyを平均して保管
                        self.entropy["training"].append(sum(train_entrpy)/len(train_entrpy))
                        #validation dataがあれば同じことを行う
                        if ((X_val is not None) & (y_val is not None)):
                                self._propagate_forward(X=X_val)
                                val_entrpy = self._calc_entropy(y=eye[y_val.reshape(-1,)])
                                self._propagate_backward(y=eye[y_val.reshape(-1,)])
                                self.entropy["validation"].append(val_entrpy)
                        
                        lap = time.time() 
                        print("epoch: ", epoch)
                        print("process time: ", lap - start, "sec")
                return self.entropy
        
        def predict(self, X):
                X = np.array(X)
                self._propagate_forward(X)
                return np.argmax(self.H2, axis=1)

        def _gen_layers(self):
                self.RNN = ScratchRNN(
                    n_nodes=self.n_nodes1,
                    initializer=HeInitializer(),
                    optimizer=AdaGrad(self.lr),
                    activator= TanH(),
                    lr = self.lr
                )
                
                self.FC = FullyConnectedLayer(
                    self.n_nodes1, self.n_output,
                    initializer=HeInitializer(), optimizer=AdaGrad(self.lr)
                )
                self.activation = Softmax()
        
        def _propagate_forward(self, X):
                H1 = self.RNN.forward(X)
                A1 = self.FC.forward(H1[:, -1, :])
                self.H2 = self.activation.forward(A1)

        def _propagate_backward(self, y):
                dA1 = self.activation.backward(self.H2, y)
                dH1 = self.FC.backward(dA1)
                dH0 = self.RNN.backward(dH1)# dH0は使用しない
        
        def _calc_entropy(self, y):
                return self.activation.cross_entropy(self.H2 , y)