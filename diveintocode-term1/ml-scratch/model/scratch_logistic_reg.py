import numpy as np
import matplotlib.pyplot as plt

class ScratchLogisticRegression():
    """
    Logistic回帰のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr: float
        学習率learning rate
    lmbd: float
        正則化パラメータ lambda
    no_bias : bool
        バイアス項を入れない場合はTrue
    unique_value: ndarray of unique element in y
        yの要素を重複なしで抜き出したndarray（yの値を0, 1に変換するために利用）
    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録

    """

    def __init__(self, num_iter, lr, lmbd, no_bias = False):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.lmbd = lmbd
        #no_bias = Trueの時はintercept = 0（切片考慮無し）に、それ以外は1に
        self.intercept = 0 if no_bias else 1
        self.unique_value = []
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        ロジスティック回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

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
        #lossの蓄積を初期化
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        #pandasの場合はnpのarrayに
        X = np.array(X)
        y = np.array(y)
        
        #XにX_0の列をintercept埋めで追加（切片の項を追加）
        X = np.concatenate((np.full((len(X), 1), self.intercept), X), axis=1)
                
        # yのshapeを(n_samples, )から(n_samples, 1)にする
        y = y.reshape(-1,1)
        
        #0, 1以外の２値で入力された場合への対応（例：a, bを0, 1に変換）
        #yの中のユニークな値を重複無しのndarrayに（[a, b, a, a, b, b] を[a, b]に ）
        self.unique_value = np.unique(y)
        #yを0, 1に変換（[a, b, a, a, b, b] がaとなる場合に0(False), bとなる場合に1(True)に ）
        y = (y != self.unique_value[0]).astype(np.int64)
        
        #validation dataに対して同様の前処理
        if not ((X_val is None) or (y_val is None)):
            X_val = np.concatenate((np.full((len(X_val), 1), self.intercept), X_val), axis=1)
            y_val = y_val.reshape(-1,1)
            y_val = (y_val != self.unique_value[0]).astype(np.int64)

        #coef_ (theta)を初期化 (1, n_features)
        #-1 ~ 1の乱数で
        a = -1
        b = 1
        self.coef_ = (b - a) * np.random.rand(X.shape[1]).reshape(1,-1)  + a     

        #self.iterの回数だけself.coef_を更新しながらhも更新していく
        for i in range(self.iter):
            #予測値h
            h = self._hypothesis(X)
            
            #hを元にしてCostの計算
            J, error = self._cost(h, y)
            
            #costをval_lossに記録
            self.loss[i] = J
            
            #validation dataに対して同様の処理
            if not ((X_val is None) or (y_val is None)):
                h_val = self._hypothesis(X_val)
                J_val, error_val = self._cost(h_val, y_val)
                self.val_loss[i] = J_val
                
            #errorを元にgradを計算してself.coef_を更新
            self._gradient_descent(X, error)
    
    def predict_proba(self, X_test):
        """
        線形回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """
        #XにX_0の列を追加（切片の項を追加）
        X_test = np.concatenate((np.full((len(X_test), 1), self.intercept), X_test), axis=1)

        #a.dot(b)よりもnp.dot(a, b)の方が早い(np.dotはC言語、a.dotはPythonでの処理なので)
        prob = self._convert_sigmoid(np.dot(X_test, (self.coef_).T))
        return prob
    
    def predict(self, X_test):
        #0に分類される確率と1に分類される確率を計算しshape(1, 2)のndarrayにする
        #さらに、確率0.5以上なら1に、0.5未満なら0に置き換える
        prob = (self.predict_proba(X_test) >= 0.5).astype(np.int64)
        
        #(0, 1)を (a, b)に戻す。0の分類の列に関して、Falseならaを、Trueならbを戻す
        pred = self.unique_value[prob[:, 0]]
        return pred

    def _convert_sigmoid(self, z):
        """
        Parameters
        -----------
        z: ndarray of shape(any, any)

        Returns
        -----------
        sigmoid_z: ndarray of shape z
            sigmoid_z = 1/(1 + exp(-z))

        Note
        -----------
        """
        sigmoid_z = 1/(1 +np.exp(-z))
        return sigmoid_z
    
    def _hypothesis(self, X):
        """
        線形の仮定関数を計算する

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
          学習データ

        Returns
        -------
          ndarray of shape (n_samples, 1)
          線形の仮定関数による推定結果

        """
        h = self._convert_sigmoid(np.dot(X, (self.coef_).T))
        
        return h
      
    def _cost(self, h, y):
        """
        costの計算

        Parameters
        ----------
        h : ndarray of shape (n_samples,)
          推定した値
        y : ndarray of shape (n_samples,)
          正解値

        Returns
        ----------
        J : np.float
        """
        
        #コストのメイン部分
        #(1/m) * sigma(-y*log(h(x)) - (1-y)*log(1-h(x)))
        J = (-np.dot(y.T, np.log(h)) -np.dot((1-y).T, np.log(1-h)))/len(y)
        
        #正則化項
        #(lambda/m)* sigma(theta^2)
        #theta = 0は除外して足し合わせる
        reg_term =  self.lmbd*np.dot(self.coef_[:, 1:], self.coef_[:, 1:].T)/len(y)
        
        #コスト（メイン + 正則化項）
        J = J + reg_term
        
        #誤差(Jの勾配の計算用に)
        error = h - y
        
        return J, error

    def _gradient_descent(self, X, error):
        """
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
          学習データ

        error: y_pred - y
           予測値と実際のyの差
        Returns
        ----------
        インスタンス変数のtheta (self.coef_)を更新

        """
        #grad = (error * X).sum(axis=0)/len(X)
        #上の計算と同義。早い方で
        
        
        #j = 0の場合
        grad_0 = np.dot(error.T, X[:, 0])/len(X)
        self.coef_[:, 0] =  self.coef_[:, 0] - self.lr * grad_0
        
        #j = 1の場合
        grad_j =  np.dot(error.T, X[:, 1:])/len(X) - self.lmbd*self.coef_[:, 1:]/len(X)
        self.coef_[:, 1:] =  self.coef_[:, 1:] - self.lr * grad_j
        """
        こちらでも同様に動く
        #一般化
        grad =  np.dot(error.T, X)/len(X) - self.lmbd*self.coef_/len(X)
        self.coef_ =  self.coef_ - self.lr * grad
        """
    #学習曲線Plot用の関数
    def plot_learning_curve(self):
        fig, ax = plt.subplots(1,1, figsize=(4,4 ))
        ax.plot(np.array(range(self.iter)), self.loss, "-", label = 'train')
        ax.plot(np.array(range(self.iter)), self.val_loss, "-", label = 'validation')

        #label
        ax.set_title('Learning Curve')
        ax.set_xlabel('n of iterations')
        ax.set_ylabel('Cost')
        ax.legend()

    #学習過程を返却
    def return_process(self):
        return self.loss, self.val_loss