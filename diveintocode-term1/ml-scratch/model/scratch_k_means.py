import numpy as np
import matplotlib.pyplot as plt
import math

class ScratchKMeans():
        def __init__(self, rpt=20, max_iter=1000, n_clusters=5):
                self.n_clusters = n_clusters
                self.max_iter=max_iter
                self.rpt = rpt
                self.n_samples = None
                self.features = None
                self.sse = None
                self.centroid = None
        
        def fit(self, X):
                X = np.array(X)
                self.n_samples = X.shape[0]
                self.n_features = X.shape[1]
                self.sse = np.inf
                
                for _ in range(self.rpt):
                        #ランダムにインデックス生成して
                        mu_idx = np.array([np.random.randint(0, self.n_samples) for i in range(self.n_clusters)])
                        #インデックスに対応するXの点をmuの初期値として割り当て
                        self.centroid = X[mu_idx]
                        cent_tmp = np.copy(self.centroid)
                        
                        #
                        loop_cnt = 0
                        while(loop_cnt <= self.max_iter):                           
                                #d: 各点X_nからmu_kまでの距離
                                #r: Xがどのクラスターに属するか
                                d = self._get_distance(X, cent_tmp)
                                r = self._get_belonging_cluster(d)
                                cent_tmp = self._update_centroid(X, d, r)
                                sse_temp = self._get_sse(d, r)
                                
                                #sseが最小となったらsse更新
                                if (self.sse > sse_temp):
                                        self.sse = np.copy(sse_temp)                    
                                        
                                #中心点が動かなくなったらループ抜ける
                                is_cent_move = ((cent_tmp - self.centroid)**(2)).sum()
                                if (is_cent_move < 1e-7):
                                        break
                                
                                #中心点更新
                                self.centroid = np.copy(cent_tmp)
                                #ループのカウント増やす
                                loop_cnt += 1
                                        
                #返り値なし
                return
        
        def predict(self, X):
                X = np.array(X)
                #distance
                d = self._get_distance(X, self.centroid)
                #cluster
                r = self._get_belonging_cluster(d)
                #コスト
                sse = self._get_sse(d, r)
                #X_nから各clusterまでの距離のうち、最小のもののクラスターのインデックス
                min_idx = d.argmin(1)

                return min_idx, d, sse

        def plot_result(self, X, cluster):
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                for i in range(self.n_clusters):
                        ax = plt.scatter(X[cluster==i, 0], X[cluster==i, 1], label=i)
                #centroidは別でプロット
                ax = plt.scatter(self.centroid[:, 0], self.centroid[:, 1], label="centroid", marker='$◯$', color='k')
                plt.show()

        def _get_norm(self, pnt, std):
                #基準点から目標点へのベクトル
                vec = pnt - std
                #ノルム
                nrm = ((vec * vec).sum(axis=1)).reshape(-1,1)
                return nrm
        
        def _get_distance(self, X, cent):
                d = np.empty((self.n_samples, 0))
                for i in range(self.n_clusters):
                        d = np.concatenate([d, self._get_norm(X, cent[i, :])], axis=1)
                return d

        def _get_belonging_cluster(self, d):
                #X_nから各clusterまでの距離のうち、最小のもののクラスターのインデックス (shape =(n_samples, ))
                min_idx = d.argmin(1)
                #cluster分類をone hot encoding
                r = np.eye(self.n_clusters)[min_idx]
                
                #こっちでもOK
                """
                r = np.zeros((self.n_samples, 0))
                for i in range(self.n_clusters):
                        r = np.concatenate([r, (min_idx == i).reshape(-1, 1)],axis=1)
                """
                return r
      
        def _update_centroid(self, X, d, r):
                cent_tmp = np.zeros((self.n_clusters, self.n_features))
                for i in range(self.n_clusters):
                        #centroidに近い点が一つもなければ最も遠い点に
                        if (r[:, i].sum(axis=0) == 0.0):
                                cent_tmp[i, :] = np.array(X[d.argmax(0)[i]])
                        else:
                                cent_tmp[i, :] = (X * r[:, i].reshape(-1,1)).sum(axis=0)/r[:, i].sum()
                return cent_tmp
        
        def _get_sse(self, d, r):
                return (d * r).sum()