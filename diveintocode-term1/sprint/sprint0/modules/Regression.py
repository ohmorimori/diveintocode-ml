import numpy as np

# Pipeline クラスをインポート
from sklearn.pipeline import Pipeline

# StanardScalerクラスをインポート
from sklearn.preprocessing import StandardScaler

# 各機械学習モデルをインポート
from sklearn.linear_model import LinearRegression

#評価のライブラリ
from sklearn.metrics import classification_report

def LinearReg(X_train, X_test, y_train, y_test):

    #パイプライン作成
    pl = Pipeline([
       ('scl', StandardScaler()),
        ('lnr', LinearRegression())
    ])

    #train with train data
    pl.fit(X_train, y_train)
    
    #Estimate test value
    pred = pl.predict(X_test)
    
    #Evaluation
    print("Score R^2", pl.score(X_train, y_train))
    
    return pred
