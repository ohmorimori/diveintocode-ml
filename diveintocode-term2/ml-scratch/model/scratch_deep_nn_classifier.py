import sys

model = "../../ml-scratch/model"
utils =  "../../ml-scratch/utils"
if model not in sys.path:
        sys.path.append(model)
if utils not in sys.path:
        sys.path.append(utils) 

from get_mini_batch import GetMiniBatch
from layer import FullyConnectedLayer
from initializer import HeInitializer
from optimizer import AdaGrad
from activator import ReLU
from activator import Softmax

import time
import numpy as np


class ScratchDeepNeuralNetrowkClassifier():
        def __init__(self, n_features = 784, n_nodes1= 400, n_nodes2 = 200, n_output = 10,  sigma=0.01, lr=0.001, batch_size=20, n_epochs=30):
                self.n_features = n_features
                self.n_nodes1 = n_nodes1
                self.n_nodes2 = n_nodes2
                self.n_output = n_output
                self.sigma = sigma
                self.lr = lr
                self.batch_size = batch_size
                self.entropy= {"training": [], "validation": []}
                self.n_epochs = n_epochs
        
        def fit(self, X, y, X_val=None, y_val=None):
                X = np.array(X)
                y = np.array(y)
                
                optimizer = AdaGrad(self.lr)
                #self.FC1 = FullyConnectedLayer(self.n_features, self.n_nodes1, SimpleInitializer(self.sigma), optimizer)
                self.FC1 = FullyConnectedLayer(self.n_features, self.n_nodes1, HeInitializer(), optimizer)
                self.activation1 = ReLU()
                self.FC2 = FullyConnectedLayer(self.n_nodes1, self.n_nodes2, HeInitializer(), optimizer)
                self.activation2 = ReLU()
                self.FC3 = FullyConnectedLayer(self.n_nodes2, self.n_output, HeInitializer(), optimizer)
                self.activation3 = Softmax()
                
                eye = np.eye(len(np.unique(y)))
                start = time.time()
                #epoch毎に
                for epoch in range(self.n_epochs):
                        #mini batch生成
                        mini_batch = GetMiniBatch(X, y, self.batch_size)

                        #mini batch毎に
                        train_entrpy = []
                        for mini_X, mini_y in mini_batch:
                                #学習(forward and/or backward propagation)
                                mini_entrpy = self._propagate(
                                        X=mini_X,
                                        y=eye[mini_y.reshape(-1,)],
                                        predict=False
                                )
                                #mini batchごとのentropy保管
                                train_entrpy.append(mini_entrpy)
                        #batchごとのentropyを平均して保管
                        self.entropy["training"].append(sum(train_entrpy)/len(train_entrpy))
                        #validation dataがあれば同じことを行う
                        if ((X_val is not None) & (y_val is not None)):
                                val_entrpy = self._propagate(
                                        X=X_val,
                                        y=eye[y_val.reshape(-1,)],
                                        predict=False,
                                        validation=True
                                )
                                self.entropy["validation"].append(val_entrpy)
                        
                        lap = time.time() 
                        print("epoch: ", epoch)
                        print("process time: ", lap - start, "sec")
                return self.entropy
        
        def predict(self, X):
                X = np.array(X)
                #forward propagationのみ
                self._propagate(X, y=None, predict=True)
                return np.argmax(self.Z3, axis=1)
        
        def _propagate(self, X, y=None, predict= False, validation=False):

                #forward propagation
                A1 = self.FC1.forward(X)
                Z1 = self.activation1.forward(A1)
                A2 = self.FC2.forward(Z1)
                Z2 = self.activation2.forward(A2)
                A3 = self.FC3.forward(Z2)
                self.Z3 = self.activation3.forward(A3)
                if (predict == True):
                        return
                #entropy
                entropy = self.activation3.cross_entropy(self.Z3 , y)     
                
                if (validation == True):
                        return entropy
                
                #backward propagation
                dA3 = self.activation3.backward(self.Z3, y)
                dZ2 = self.FC3.backward(dA3)
                dA2 = self.activation2.backward(dZ2)
                dZ1 = self.FC2.backward(dA2)
                dA1 = self.activation1.backward(dZ1)
                dZ0 = self.FC1.backward(dA1) # dZ0は使用しない
                
                return entropy