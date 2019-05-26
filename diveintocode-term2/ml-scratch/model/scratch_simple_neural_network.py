import numpy as np
import time
class Layer():
        """
        ニューラルネットワーク分類器の各層を表すクラス

        Parameters, Attributes
        ----------
        self.layer: int(1 to n)
            自身が全体の何層目かを表す
        self.node_size_list: list 
            node sizeのリスト node[layer]でそのlayerのnode_sizeを取得
        self.feature: ndarray of shape(batch_size, n_nodes)
            前の層から渡されるfeature
        self.target: ndarray of shape(batch_size, )
           target値
        self.lr: float
            learning rate
        self.max_layer: int
            len(node_size_list)
            最後の層を表す整数。再帰を止めるために用意 
        self.n_nodes: int
            そのlayerのノード数
            node_size_list[layer - 1]
        self.n_features: 
            そのlayerのfeature数。pre_layerのnode数に一致
        self.pre_layer: Layerクラスのインスタンスへの参照
            1つ前の層を表す
        self.nxt_layer: Layerクラスのインスタンスへの参照
            1つ後の層を表す
        self.coef_: ndarray of shape(pre_layer.n_node, current_layer.n_node)
            coefficient W
        self.bias : ndarray of shape(1, current_layer.n_node)
            coefficient B
        self.a: ndarray of shape(batch_size, current_layer.n_node)
            A = pre_layer.Z @ W + B
        self.z: ndarray of shape(batch_size, current_layer.n_node)
            Z = activator(A)
        self.dL_dA: ndarray of shape(1, current_layer.n_node)
            gradient of L to A
        self.dL_dB: ndarray of shape(1, current_layer.n_node)
            gradient of L to B
        self.dL_dW: ndarray of shape(pre_layer.n_node, current_layer.n_node)
            gradient of L to W
        self.dL_dZ = ndarray of shape(1, current_layer.n_node)
            gradient of L to Z
        self.activator: string
            activatorの関数の種類 ('tanh' or 'sigmoid') 
        self.sigma: float
            sigma to initialize Layer class coef
        """

        
        def __init__(self, pre_layer, layer, node_size_list, feature=None, target=None, lr=0.001, activator='tanh', sigma=0.01):
                self.layer = layer
                self.node_size_list = node_size_list
                self.feature = feature
                self.target = target
                self.lr = lr
                self.max_layer = len(node_size_list)
                self.n_nodes = node_size_list[layer - 1]
                self.n_features = None
                self.pre_layer = pre_layer
                self.nxt_layer = None
                self.coef_ = None#W
                self.bias = None#B
                self.a = None
                self.z = None
                self.dL_dA = None
                self.dL_dB = None
                self.dL_dW = None
                self.dL_dZ = None
                self.activator = activator
                self.sigma = sigma
        
        def _init_coef(self):
                #係数の初期化がされている時は
                if ((self.coef_ is not None) | (self.bias is not None)):
                        #実行しない
                        return
                #初期化がされていない時は初期化
                self.n_features = self.feature.shape[1]
                #w
                if (self.layer > 1):
                        self.coef_ =  self.sigma * np.random.randn(self.pre_layer.n_nodes, self.n_nodes)
                #1層目の時はpre_layerないのでこっち
                else:
                        self.coef_ =  self.sigma * np.random.randn(self.n_features, self.n_nodes)
                #b
                self.bias = self.sigma * np.random.randn(self.n_nodes,)
                
                print("new layer generated!", self.layer)
        
        def _gen_next_layer(self):
                self.nxt_layer = Layer(
                        pre_layer=self,
                        layer=self.layer + 1,
                        node_size_list= self.node_size_list,
                        feature=self.z,
                        target=self.target,
                        lr=self.lr,
                        activator=self.activator,
                        sigma = self.sigma
                )
                return
                 
        def propagate_forward(self, feature, target):
                #初期化する必要があればcoefを初期化
                if ((self.coef_ is None) | (self.bias is None)):
                        self._init_coef()
                #aを計算
                self.a = feature @ self.coef_ + self.bias

                #zを計算
                feature = self._activate_forward(self.a)
                self.z = feature
                
                #最後の層に達したら
                if (self.layer == self.max_layer):
                        #自分自身を返して再帰終了
                        return self
                
                #最後の階層でなければ
                if (self.layer != self.max_layer):
                        #次のlayerが生成されていない時のみ
                        if(self.nxt_layer is None):
                                #次のlayerのインスタンスを生成
                                self._gen_next_layer()
                        #各層のtargetを更新しておく(backwardのためにlast_layerに伝達しておく)
                        self.nxt_layer.target = target
                
                #次の層を呼んで再帰
                return self.nxt_layer.propagate_forward(feature=feature, target=target)
        
        def propagate_backward(self):
                #dL_dAの算出
                if (self.layer == self.max_layer):
                        #最後の層の場合
                        self.dL_dA = np.mean(self.z - self.target, axis=0).reshape(1, -1)#sample size方向に平均   
                else:
                        self.dL_dA = (self.dL_dZ * self._activate_backward(np.mean(self.a, axis=0).reshape(1, -1)))
                
                #dL_dWとdL_dZの算出
                if (self.layer ==1):
                        #最初の層の場合 (pre_layerが存在しないので算出できない)
                        self.dL_dW = np.mean(self.feature.T, axis=1).reshape(-1, 1) @ self.dL_dA#sample size方向に平均 
                else:
                        self.dL_dW = np.mean(self.pre_layer.z.T, axis=1).reshape(-1, 1) @ self.dL_dA#sample size方向に平均   
                        self.pre_layer.dL_dZ = self.dL_dA @ self.coef_.T

                #係数更新
                self._update_coef()
                
                #一層目に達したら
                if (self.layer == 1):
                        #自分自身をを返却して終了
                        return self
                #一層目まで達していなかったら
                #１つ前の層を呼んで再帰
                return self.pre_layer.propagate_backward()
        
        def _update_coef(self):
                self.coef_ = self.coef_ - self.lr * self.dL_dW
                self.bias = self.bias - self.lr * self.dL_dA
                return
        
        def _tanh(self, a):
                #(np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a))
                #式変形してこっち
                return 1 - 2 / (1+ np.exp(2 * a))
            
        def _sigmoid(self, a):
                return 1/(1 + np.exp(-a))
            
        def _softmax(self, a):
                #np.exp(a)/np.exp(a).sum(axis=1, keepdims=True)
                #まともに計算してしまうと分母分子がそれぞれe^300を超えてオーバーフローする
                #求めたいのは比なので、e^(a+x)/e^(b+x) = e^a * e^x/e^b +e^x =  e^(a-b)とすれば良い。
                #よって、特徴両方向に一律にaのmax値を引いておいてから計算する
                e_a = np.exp(a - np.max(a, axis=1, keepdims=True))
                return e_a/e_a.sum(axis=1, keepdims=True)
            
        def _activate_forward(self, a):
                #最後の層なら
                if (self.layer == self.max_layer):
                        return self._softmax(a)
                #最後の層以外なら
                elif(self.activator == 'sigmoid'):
                        return self._sigmoid(a)
                elif(self.activator == 'tanh'):
                        return self._tanh(a)
        
        def _activate_backward(self, a):
                #最後の層なら
                if (self.layer == self.max_layer):
                        return
                #最後の層以外なら
                elif(self.activator == 'sigmoid'):
                        #sigmoidの一階微分
                        return (1 - self._sigmoid(a)) * self._sigmoid(a)
                elif(self.activator == 'tanh'):
                        return (1 - self._tanh(a)**2)
        
        def get_entropy(self, target, proba):
                return - (target * np.log(proba)).sum() / target.shape[0]

class ScratchSimpleNeuralNetrowkClassifier():
        """
        シンプルな三層ニューラルネットワーク分類器

        Parameters, Attributes
        ----------
        self.verbose: bool
            trueなら学習のコストを返却
        self.node_size_list: list of shape(number of layers, )
            各layerのnodeのsizeを記録したlist
        self.batch_size: int
            batch_size
        self.lr: float
            学習率
        self.activator: string
            activatorの関数の種類 ('tanh' or 'sigmoid') 
        self.first_layer: Layerクラスのインスタンス
            1層目
        self.last_layer: Layerクラスのインスタンス
            最後の層
        self.n_epochs: int
            number of epochs
        self.cost: dictionary of shape = (n_epochs, 2)
            {"training": [], "validation": []}
        self.sigma: float
            sigma to initialize Layer class coef
        

        """

        def __init__(self, verbose=True, batch_size=10, node_size_list=[400, 200, 10], activator='tanh', lr=0.001, n_epochs=100, sigma=0.01):
                self.verbose = verbose
                self.node_size_list = node_size_list
                self.batch_size = batch_size
                self.lr = lr
                self.activator = activator
                self.first_layer = None
                self.last_layer = None
                self.n_epochs = n_epochs
                self.cost = {"training": [], "validation": []}
                self.sigma = sigma
                
        def fit(self, X, y, X_val=None, y_val=None):
                """
                ニューラルネットワーク分類器を学習する。

                Parameters
                ----------
                X : 次の形のndarray, shape (n_samples, n_features)
                    学習用データの特徴量
                y : 次の形のndarray, shape (n_samples, )
                    学習用データの正解値
                X_val : 次の形のndarray, shape (n_samples, n_features)
                    検証用データの特徴量
                y_val : 次の形のndarray, shape (n_samples, )
                    検証用データの正解値
                """
                
                #X, y整形
                X = np.array(X)
                y = np.array(y).reshape(-1, 1)
                
                #1層目を作る
                self.first_layer = Layer(
                        pre_layer=None,
                        layer=1,
                        node_size_list=self.node_size_list,
                        lr=self.lr,
                        activator=self.activator,
                        sigma=self.sigma
                )
                
                start = time.time()
                
                #target trainのone-hot vector得るためにtargetのuniqueなlabelを分の単位行列
                eye = np.eye(len(np.unique(y)))
                
                for epoch in range(self.n_epochs):
                        mini_batch = GetMiniBatch(X, y, self.batch_size)
                        train_entropy = []
                        for mini_X, mini_y in mini_batch:
                                #1層目にmini batchのfeatureとtargetを渡す
                                self.first_layer.feature = mini_X
                                self.first_layer.target = eye[mini_y.reshape(-1,)]#one hot化したyをtargetに        

                                #forward & backward propagationしてentropy算出
                                mini_entropy = self._propagate()                                
                                #mini_batchのentropy記録
                                train_entropy.append(mini_entropy)                               

                        #epochごとのmini_batchのentropy平均を格納
                        self.cost["training"].append(sum(train_entropy)/len(train_entropy))
                                
                        #validation dataがあればforward propagationのみ実行してentropy計算
                        if ((X_val is not None) & (y_val is not None)):
                                #forward propagation
                                self.first_layer.propagate_forward(
                                    feature=X_val,
                                    target=eye[y_val.reshape(-1,)]
                                )
                                #entropy
                                val_entropy = self.last_layer.get_entropy(
                                        target=eye[y_val.reshape(-1,)],
                                        proba=self.last_layer.z
                                )
                                self.cost["validation"].append(val_entropy)
                        
                        #lap timeとサイクル数表示
                        lap = time.time() 
                        print("epoch: ", epoch)
                        print("process time: ", lap - start, "sec")
                
                if self.verbose:
                    #verboseをTrueにした際は学習過程を出力する
                    return self.cost
                
                return
        
        def predict(self, X):
                """
                ニューラルネットワーク分類器を使い推定する。

                Parameters
                ----------
                X : 次の形のndarray, shape (n_samples, n_features)
                    サンプル

                Returns
                -------
                    次の形のndarray, shape (n_samples, 1)
                    推定結果
                """
                X = np.array(X)
                
                self.first_layer.feature = X
                self.first_layer.propagate_forward(feature=X, target=None)  
                #最後の層のzがyの各ラベルに対応した確率
                prob = self.last_layer.z
                #確率が最大となるラベルのインデックスを取得(今回の場合はインデックスそのものの値がラベル) 
                pred = np.argmax(prob, axis=1)                       
                return pred
            
        def _propagate(self):
                #2層目以降をつくりながら、再帰的にcoef更新していく
                #forward_propagation
                self.last_layer = self.first_layer.propagate_forward(
                        feature=self.first_layer.feature,
                        target=self.first_layer.target
                )

                #back_propagation
                self.first_layer = self.last_layer.propagate_backward()#n -> 1層目へ                                
                
                #entropy計算
                mini_entropy = self.last_layer.get_entropy(
                        target=self.last_layer.target,
                        proba=self.last_layer.z
                )
                
                return mini_entropy    
        
class GetMiniBatch:
        """
        ミニバッチを取得するイテレータ

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        y : 次の形のndarray, shape (n_samples, 1)
          正解値
        batch_size : int
          バッチサイズ
        seed : int
          NumPyの乱数のシード
        """
        def __init__(self, X, y, batch_size=10, seed=0):
                self.batch_size = batch_size
                np.random.seed(seed)
                shuffle_index = np.random.permutation(np.arange(X.shape[0]))
                self.X = X[shuffle_index]
                self.y = y[shuffle_index]
                self._stop = np.ceil(X.shape[0]/self.batch_size).astype(np.int)

        def __len__(self):
                return self._stop

        def __getitem__(self,item):
                p0 = item*self.batch_size
                p1 = item*self.batch_size + self.batch_size
                return self.X[p0:p1], self.y[p0:p1]        

        def __iter__(self):
                self._counter = 0
                return self

        def __next__(self):
                if self._counter >= self._stop:
                        raise StopIteration()
                p0 = self._counter*self.batch_size
                p1 = self._counter*self.batch_size + self.batch_size
                self._counter += 1
                return self.X[p0:p1], self.y[p0:p1]