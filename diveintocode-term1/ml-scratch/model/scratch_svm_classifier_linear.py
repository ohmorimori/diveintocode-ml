import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

class ScratchSVMClassifier_linear():
        """
        Support Vector Machineのスクラッチ実装

        Parameters
        -------------
        num_iter: int
            反復数
        lr: float
            Learning rate

        Attributes
        -------------
        coef_: ndarray, shape = [n_features, ]
        cost: ndarray, shape = [self._iter, ]
        lmbd: ndarray, shape = [n_samples, ]
            ラグランジュ乗数
        """
        def __init__(self, num_iter=500, lr = 10**(-5), lmbd_threshold = 10**(-4), no_bias = False):
                self.iter = num_iter
                self.lr = lr
                #no_bias = Trueの時はintercept = 0（切片考慮無し）に、それ以外は1に
                self.intercept = 0 if no_bias else 1
                # 目的関数を記録する配列を用意
                self.cost = np.zeros(self.iter)
                self.lmbd_threshold = lmbd_threshold
                #以下、fit時に初期化するものをとりあえず空で初期化

                #サンプル数を保管
                self.n_samples = None
                #ラグランジュ乗数
                self.lmbd = None
                #support vector参照用にtrainデータをインスタンスに保管する
                self._train_X = None
                self._train_y = None
                #support vectorのインデックス
                self.sv_idx = None
                #係数
                self.coef_  = None
                self.coef_0 = None

                #yの重複無しの要素を格納
                self.unique_value = None
                
        def fit(self, X, y, X_val=None, y_val=None):
                """
               SVMで学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

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
                #X, yをnpのarrayにしておく
                X = np.array(X)
                y = np.array(y)

                #train dataをfit後にも参照できるように保管しておく
                self._train_X = np.copy(X)
                self._train_y = np.copy(y)
                
                #サンプル方向を縦に
                y = y.reshape(-1, 1)

                #-1, 1以外の２値で入力された場合への対応（例：a, bを-1, 1に変換）

                #yの中のユニークな値を重複無しのndarrayに（[a, b, a, a, b, b] を[a, b]に ）
                self.unique_value = np.unique(y)
                #yを0, 1に変換
                #（[a, b, a, a, b, b] がaとなる場合に0(False), bとなる場合に1(True)に ）
                y = (y != self.unique_value[0]).astype(np.int64) 
                #yを-1, 1に変換
                y = (1-(-1))*y - 1 

                #サンプル数をインスタンス変数に保管
                self.n_samples = len(X)
                #サンプル数分のラグランジュ乗数をゼロで初期化
                self.lmbd = np.zeros(self.n_samples).reshape(-1, 1)

                #lmbdを算出
                self._update_lmbd(X, y)

                #lmbdを元にしてtheta(coef_)を算出
                self._gradient_descent(X, y)

        def predict(self, X_test):
                X_test = np.array(X_test)

                pred = np.dot(self.coef_, X_test.T) + self.coef_0
                #0より大きければ1, 小さければ0に変換
                pred = (pred > 0).astype(np.int64)
                #0ならa, 1ならbに変換
                pred = self.unique_value[pred]
                return pred

        def _update_lmbd(self, X, y):
                """
                Parameters
                -----------
                X: ndarray of shape (n_samples, n_features)
                y: ndarray of shape (n_samples, 1)

                Returns
                -----------
                none

                Note
                -----------
                update instance variable of lagrange coefficient lambda

                """
                for i in range(self.iter):
                        L_grad = np.ones(self.n_samples).reshape(-1, 1) - ((y * X) @ (X.T @ (self.lmbd * y)))
                        self.lmbd = self.lmbd + self.lr * (L_grad)

                        #iterごとのコストを保管
                        self.cost[i] = self._L(X, y)

                #return値無し
                return

        def _L(self, X, y):
                return  self.lmbd.sum() - (((self.lmbd * y).T @ X) @ (X.T @ (self.lmbd * y)))/2

        def _gradient_descent(self, X, y):
            #lambda > 0を満たしているものをサポートベクターとしてTrueでラベルしてインデックスに使用
            #サポートベクトルを絞るため、便宜上lmbd = 0.0001などにする
            self.sv_idx = (self.lmbd > self.lmbd_threshold).reshape(-1,)
            #thetaの算出
            self.coef_ = (self.lmbd[self.sv_idx] * y[self.sv_idx]).T @ X[self.sv_idx, :]
            #theta_0の算出
            self.coef_0 =  (y[self.sv_idx] - (X[self.sv_idx] @ self.coef_.T )).sum(axis=0)/(self.sv_idx).sum()
            
            #return値無し
            return

        def decision_region(self, X, y, step=0.01, title='decision region', xlabel='xlabel', ylabel='ylabel',  target_names=['1', '-1']):
                """
                ２値分類を２次元の特徴量で学習したモデルの決定領域を描く
                背景の色が学習したモデルによる推定値から描画される
                散布図の点は学習用のデータである。

                Parameters
                ---------------
                X: ndarray, shape(n_samples, 2)
                    特徴量
                y: ndarray, shape(n_samples,)
                    正解値
                model: object
                    学習したモデルのインスタンスを入れる
                step: float, (default: 0.1)
                    推定値を計算する間隔を設定する
                title: str
                    グラフのタイトルの文章を与える
                xlabel, ylabel: str
                    軸ラベルの文章を与える
                target_names=: list of str
                    凡例の一覧を与える
                ---------------
                """
                #setting
                scatter_color = ['red', 'blue']
                contourf_color = ['skyblue', 'pink']
                n_class = 2


                #pred
                #各特徴量に対してメッシュを生成（a=(a1, a2, a3), b=(b1, b2, b3)に対して[[a1, b1], [a1, b2], [a1, b3]], [[a2, b1], [a2, b2], [a2, b3]], [[a3, b1], [a3, b2], [a3, b3]]を生成）
                mesh_f0, mesh_f1 = np.meshgrid(np.arange(np.min(X[:,0])-0.5, np.max(X[:,0])+0.5, step), np.arange(np.min(X[:,1])-0.5, np.max(X[:,1])+0.5, step))

                #多次元を一次元配列に
                mesh = np.c_[np.ravel(mesh_f0),np.ravel(mesh_f1)]
                pred = self.predict(mesh).reshape(mesh_f0.shape)

                #plot
                fig, ax = plt.subplots(1,1, figsize=(6,4 ))
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                #等高線塗りつぶし
                ax.contourf(mesh_f0, mesh_f1, pred, n_class-1, cmap=ListedColormap(contourf_color))
                #等高線塗りつぶし
                ax.contour(mesh_f0, mesh_f1, pred, n_class-1, colors='y', linewidths=3, alpha = 0.5)
                #重複しない要素に対して, 答えをプロット
                for i, target in enumerate(set(np.unique(y))):
                    ax.scatter(X[y==target][:, 0], X[y==target][:, 1], s=20, color=scatter_color[i], label=target_names[i], marker='o')

                #trainデータに関してプロットしている時には
                if (self._check_same_ary(y, self._train_y)):
                    #support vectorを黄色に
                    ax.scatter(X[self.sv_idx, 0], X[self.sv_idx, 1], s=20, color='yellow' , label='supprt vector', marker='o')

                ax.legend()


                #return値無し
                return

        def _check_same_ary(self, a, b):
                return (a == b).all() if (a.shape == b.shape) else False

        #学習曲線Plot用の関数
        def plot_learning_curve(self):
                fig, ax = plt.subplots(1,1, figsize=(4,4 ))
                ax.plot(np.array(range(self.iter)), self.cost, "-", label = 'train')

                #label
                ax.set_title('Learning Curve')
                ax.set_xlabel('n of iterations')
                ax.set_ylabel('Lagrangian')
                ax.legend()

                #return値無し
                return