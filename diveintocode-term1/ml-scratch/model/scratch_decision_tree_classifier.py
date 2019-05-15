import numpy as np

class Node():
        """
        Note
        --------------
        Node class of decision tree

        Parameters
        --------------

        Attributes
        --------------
        self.criterion: string ('gini' or 'entropy')
                to decide which criteria to use for calculating info gain
        self.max_depth: int.
                max depth of tree from root node
        self.random_state: 
        self.depth: int.
                depth from root node (flag for search stop at designated max_depth)
        self.left: Node.
                left child
        self.right: Node.
                right child
        self.feature: int (index of columns of a feature)
                a feature with which to classify
        self.threshold: int.
                a threshold with which to classify
        self.label: 
                a classified label judged by the threshold
        self.impurity: float
                gini impurity
        self.info_gain: float
                information gain
        self.n_samples: int.
                number of samples
        self.n_classes: int.
                number of classes in the feature of interest 
        """

        def __init__(self, criterion='gini', max_depth=None, random_state=None):
                self.criterion = criterion
                self.max_depth = max_depth
                self.random_state = random_state
                self.depth = None
                self.left = None
                self.right = None
                self.feature = None
                self.threshold = None
                self.label = None
                self.impurity = None
                self.info_gain = None
                self.n_samples = None
                self.n_classes = None

        def _split_node(self, X, y, depth, ini_n_classes):
                #pandas -> np array
                X = np.array(X)
                y = np.array(y)

                self.depth = depth
                self.n_samples = len(y)
                self.n_classes = [len(y[y == i]) for i in ini_n_classes]

                #各クラスのカウント数をdict{key=class名, value=count数}に
                class_count = {i: len(y[y == i]) for i in np.unique(y)}
                #カウント数の最大値のラベルを取得
                self.label = max(class_count.items(), key=lambda x:x[1])[0]
                """
                Memo
                -------
                lambdaは無名関数。lambda a:bで受け取った各要素aに対してbを行う
                ここでは要素xを受け取り、x[1]を返すという意味。
                辞書class_countのアイテム(keyとvalueのセット)を取り出して、
                value(x[1])に関してmax値を求め、そのvalueとkeyをセットとして返している
                """
                self.impurity = self._criterion_func(y)
                
                #要素数を取得
                n_features = X.shape[1]
                #info_gainの初期値
                self.info_gain = 0.0
                
                #葉に到達したら
                #特徴量のクラスが一つだけなら全データが同一クラスとなった(葉に到達)ということ
                if (len(np.unique(y)) == 1):
                        #探索終了
                        return
                    
                #節だったら（葉に到達していなかったら）探索継続
                
                #random_stateが設定されていたら
                if (self.random_state != None):
                        np.random.seed(self.random_state)
                
                #0 - n_featuresのランダムな配列
                #特徴量が選ばれる順番によるバイアスをなくすため
                f_loop_order = np.random.permutation(n_features).tolist()

                """
                permutation(x)はxのコピーを並び替え、shuffle(x)はin-placeで並び替える
                ここでは0-xまでの連続したintを生成してランダムに並び替えて返している
                """

                #各特徴量に関して
                for f in f_loop_order:
                        #その特徴量内の固有な値を抜き出し
                        uniq_feature = np.unique(X[:, f])
                        #分割点を決める（最初と最後の点は除いて平均を出す）
                        split_points = (uniq_feature[:-1] + uniq_feature[1:])/2.0

                        """
                        a = np.array(['b','a','b','b','d','a','a','c','c'])のとき
                        np.unique(a)は順序を昇順にしてユニークな要素を返す['a', 'b', 'c']
                        元の順番を守って['b','a','c']と返したい時はnp.unique(a, return_index=True)
                        """
                        #各thresholdに対して
                        for threshold in split_points:
                                #targetをthresholdに基づいてleft, rightに分割
                                y_l = y[X[:, f] <= threshold]
                                y_r = y[X[:, f] > threshold]

                                #分割によるinfo gainを計算
                                val = self._calc_info_gain(y, y_l, y_r)

                                #保持していたinfo_gainと比較して、計算値が大きければ
                                if (self.info_gain < val):
                                        #新たにnfo_gainとして保管し、その時のfeatureとthresholdも保管
                                        self.info_gain = val
                                        self.feature = f
                                        self.threshold = threshold

                #探索の終了判定
                
                #info gainがゼロなら終わり
                if (self.info_gain == 0.0):
                        return
                #depthが初期化時に引数で指定していたmax_depthに達していたら終わり
                if (depth == self.max_depth):
                        return

                #探索終わっていなかったら再帰的に呼ぶ
                #左側
                X_l = X[X[:, self.feature] <= self.threshold]
                y_l = y[X[:, self.feature] <= self.threshold]
                self.left = Node(self.criterion, self.max_depth)
                self.left._split_node(X_l, y_l, depth + 1, ini_n_classes)

                #右側
                X_r = X[X[:, self.feature] > self.threshold] 
                y_r = y[X[:, self.feature] > self.threshold]
                self.right = Node(self.criterion, self.max_depth)
                self.right._split_node(X_r, y_r, depth + 1, ini_n_classes)

        def _criterion_func(self, y):
                """
                calculate impurity
                """
                classes = np.unique(y)
                n_data = len(y)
                
                #gini_impurity、entropyはどちらも減少するほど乱雑さが少なくなる
                #gini_impurityを基準にする時
                if (self.criterion == "gini"):
                        val = 1
                        for cls in classes:
                                p = float(len(y[y == cls]) / n_data)
                                val -= p ** 2.0
                #entropyを基準にする時
                elif self.criterion == "entropy":
                        val = 0
                        for cls in classes:
                                p = float(len(y[y == cls]) / n_data)
                                if p != 0.0:
                                        val -= p * np.log2(p)
                return val
                
        def _calc_info_gain(self, y_parent, y_left, y_right):
                """
                calculate infomation gain
                """
                #まずジニ不純度 or エントロピーを算出
                #親
                cri_p = self._criterion_func(y_parent)
                #左の子
                cri_l = self._criterion_func(y_left)
                #右の子
                cri_r = self._criterion_func(y_right)

                #親から見た左の子の比率
                l_ratio = len(y_left) / len(y_parent)
                #親から見た右の子の比率
                r_ratio = len(y_right) / len(y_parent)

                #information gainを算出
                info_gain =  cri_p - (l_ratio * cri_l + r_ratio * cri_r)
                return info_gain

        def _predict(self, X_test):
                """
                ラベルを予測して返す
                """
                #葉の場合か、指定した深さの節に達した場合
                if (self.feature == None or self.depth == self.max_depth):
                    #自身のラベルを返す
                    return self.label    
                #節の場合は左右の子に分けて探索継続
                else:
                        #Xがthreshold以下ならば左側、
                        if (X_test[self.feature] <= self.threshold):
                                return self.left._predict(X_test)
                        #Xがthresholdより大きければ右側
                        else:
                                return self.right._predict(X_test)
        
    
class ScratchDecisionTreeClassifier():
        def __init__(self, criterion="gini", max_depth=None, random_state=None):
                self.criterion = criterion
                self.max_depth = max_depth
                self.random_state = random_state
                self.tree_analysis = TreeAnalysis()
                self.root = None
                self.feature_importances_ = None
        
        def fit(self, X, y):
                """
                Train dataで学習
                """
                #Nodeクラスのインスタンスを生成しroot nodeとする
                self.root = Node(self.criterion, self.max_depth, self.random_state)
                #rootを分割してtreeを構成していく
                self.root._split_node(X, y, 0, np.unique(y))
                #feature_importance
                self.feature_importances_ = self.tree_analysis._get_feature_importance(self.root, X.shape[1])

        def predict(self, X_test):
                """
                学習結果を用いてテストデータ予測
                """
                pred = []
                
                #ここはforじゃないとダメ?
                for s in X_test:
                        pred.append(self.root._predict(s))
                
                #np.arrayに変換して返却
                return np.array(pred)
        
        def score(self, X_test, y_test):
                return sum(self.predict(X_test) == y_test)/float(len(y_test))
                
class TreeAnalysis():
        def __init__(self):
                self.n_features = None
                self.importances = None
        
        def _compute_feature_importance(self, node):
                #葉なら何もしない
                if (node.feature == None):
                        return
                
                #節なら
                #featureごとの重要度を計算して足し合わせていく
                self.importances[node.feature] += node.info_gain * node.n_samples
                
                #左右の子に対して再帰
                self._compute_feature_importance(node.left)
                self._compute_feature_importance(node.right)
        
        def _get_feature_importance(self, root, n_features, normalize=True):
                self.n_features = n_features
                self.importances = np.zeros(n_features)
                
                self._compute_feature_importance(root)
                self.importances /= root.n_samples
                
                
                if (normalize):
                        normalizer = np.sum(self.importances)
                        #ゼロで割らないような対策
                        #(rootの要素が一つしかないときなどは)
                        if (normalizer > 0.0):
                                self.importances /= normalizer
                
                return self.importances
                        