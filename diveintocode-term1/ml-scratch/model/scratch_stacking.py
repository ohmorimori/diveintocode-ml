import numpy as np

#学習モデルをインポート
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

class StackingLayer():
        """
        Note
        --------------------------------
        stackingにおける各stageのクラス
        各ステージにおけるlearnとestimateを担う

        Parameters, Attributes
        --------------------------------
        max_stage: int
            最後のステージ（Layer）。再帰を止めるために用意
        stage: int
            現在のステージ（Layer）。再帰を止めるために用意
        n_train_samples: int
            分割前のtrainサンプルのサイズ
        K: int
            そのステージにおけるtrainデータの分割数
        M: int
            そのステージにおけるモデルの数（種類数）
        model_instance：nparray shape(M, K)
            modelのインスタンスを格納
        nxt_layer: StackingLayerクラスのインスタンス
            一つ後のLayer（ステージ）。再帰的に呼ぶために用意
        """
        def __init__(self, max_stage, stage):
                self.max_stage = max_stage
                self.stage  = stage
                self.n_train_samples = None 
                self.K = None
                self.M = None
                self.model_instance = None
                self.nxt_layer = None
        
        #学習時
        def learn(self, split_list, model_list, feat_train, target_train, ):
                #KとMを記録
                self.K = len(split_list)
                self.M = len(model_list[self.stage])
                #サンプル数を記録
                self.n_train_samples = target_train.shape[0]
                """
                Split array
                """
                #train dataをtrainとvalidationにさらにK分割するために、それぞれのindexを生成
                idx_train, idx_valid = self._gen_split_index()
                
                """
                fit & pred to generate blend data
                """
                #blend_dataを空で初期化
                blend_data =  np.empty((self.n_train_samples, self.M), dtype='float64')
                #モデルのインスタンスの格納先を空で初期化
                self.model_instance = np.empty((self.M, self.K), dtype='object')
                
                #モデル数に対してループ
                for i in range(self.M):
                        #分割数に対してループ
                        for j in range(self.K):
                                #モデルのインスタンスを生成して格納
                                self.model_instance[i, j] = self._gen_model_instance(model_list[self.stage][i])            
                                #fit & predict
                                #train側に分けたfeatureとtargetデータでfit
                                self.model_instance[i, j].fit(feat_train[idx_train[:, j]], target_train[idx_train[:, j]])
                                #valid側に分けたfeatureデータでpredict
                                pred = self.model_instance[i, j].predict(feat_train[idx_valid[:, j]]).reshape(-1,)
                                
                                #prediction dataをblend
                                #k=Nの時は要らないのでスキップ
                                if (self.stage != self.max_stage): 
                                        blend_data[idx_valid[:, j], i] =  pred
                """
                Next Stage
                """
                #k=Nになるまで再帰させる
                
                #stageがk = Nなら
                if (self.stage == self.max_stage):
                        #再帰終了
                        return 
                
                #stage 0 to N-1の時
                #next_layerのインスタンス生成
                self.nxt_layer = StackingLayer(
                        max_stage=self.max_stage,
                        stage=self.stage + 1
                )
                
                #子のステージの学習
                self.nxt_layer.learn(
                        split_list=split_list,
                        model_list=model_list,
                        feat_train=blend_data,
                        target_train=target_train
                )
                return
        
        #推定時
        def estimate(self, feat_test):
                #testのサンプル数記録
                n_test_samples = feat_test.shape[0]

                #stage Nの時
                if (self.stage == self.max_stage):
                        #予測値を返却して再帰終了
                        return self.model_instance[0, 0].predict(feat_test)
                #stage 0 to N-1の時                
                """
                predict with learned model
                """
                #blend testの格納先を初期化
                blend_test =  np.empty((n_test_samples, self.M), dtype='float64')
                
                for i in range(self.M):
                        #各モデルによるpredの一時的保管のために用意
                        pred = np.empty((n_test_samples, 0), dtype='float64')
                        for j in range(self.K):
                                pred = np.append(pred, self.model_instance[i][j].predict(feat_test).reshape(-1, 1), axis=1)
                         #分割数方向に平均をとって、shape(n_test_samples, model数)のブレンドテストを生成
                        blend_test[:, i] =  np.mean(pred, axis=1)
                """
                Next Stage
                """
                #k=Nになるまで再帰
                return self.nxt_layer.estimate(blend_test)
        
        def _gen_split_index(self):
                #Trainとvalidationデータのインデックス格納のための配列を初期化
                idx_valid = np.empty((self.n_train_samples, 0), dtype='bool')
                
                #ランダムなインデックスを生成
                rand_idx= np.arange(0, self.n_train_samples)
                np.random.shuffle(rand_idx)

                for i in range(self.K):
                        #validationデータのインデックスを生成
                        idx = rand_idx[round(self.n_train_samples * (i / self.K)):round(self.n_train_samples * ((i + 1) / self.K))]
                        #数字の羅列を、対応するインデックスがTrueになった配列に変換
                        idx = self._convert_index_to_bool_array(idx, self.n_train_samples).reshape(-1, 1)
                        #ndarrayに格納
                        idx_valid = np.append(idx_valid, idx, axis=1).astype(bool)

                #validationのbool反転がtrainデータに対応
                idx_train =  ((-1) * idx_valid + 1).astype(bool)
                
                #K=1の場合（分割しない場合)
                if (self.K == 1):
                        #idx_trainが全てFalse、idx_validが全てTrueとなってしまい学習できないので、両者とも全てTrue
                        idx_train = idx_valid
                return idx_train, idx_valid
                
        def _convert_index_to_bool_array(self, idx, size):
                #数字をTrue, Falseのindexで取得する関数
                bl =  np.zeros(size, dtype=bool)
                bl[idx] = True  
                return bl
        
        def _gen_model_instance(self, model_name):
                if (model_name == 'lnr'):
                        return LinearRegression()
                elif (model_name == 'dtr'):
                        return DecisionTreeRegressor()
                elif (model_name == 'svr'):
                        return SVR()
                
class ScratchStacking():
        """
        Note
        --------------------------------
        stackingにおける全stageのまとまりを表現したクラス

        Parameters
        --------------------
        root: object
            ステージ0のStackingLayer
            
        Attributes
        --------------------------------
        split_list: list shape(n of stages,)
            各ステージにおける
        model_list: list shape(n of models, )
            各ステージにおいて使うモデルのリスト
        """
        def __init__(self):
                self.root = None

        def fit(self, split_list, model_list, feat_train, target_train): 
                #データ整形
                feat_train = np.array(feat_train)
                target_train = np.array(target_train).reshape(-1, 1)
                
                #stage 0のインスタンスを生成
                self.root = StackingLayer(
                        max_stage=len(split_list) - 1,
                        stage=0
                )
                
                #学習
                self.root.learn(
                        split_list=split_list,
                        model_list=model_list,
                        feat_train=feat_train,
                        target_train=target_train
                )
                #返却値なし
                return
        
        def predict(self, feat_test):
                #データ整形
                feat_test = np.array(feat_test)
                #推定
                pred = self.root.estimate(feat_test)
                return pred
        
        def evaluate(self, target_test, pred):
                #データ整形
                target_test = np.array(target_test).reshape(-1, 1)
                #MSE算出
                mse = ((target_test - pred).T @ (target_test - pred)/len(target_test))[0][0]
                return mse