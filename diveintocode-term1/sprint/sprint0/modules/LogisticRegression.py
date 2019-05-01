import numpy as np

# Pipeline クラスをインポート
from sklearn.pipeline import Pipeline

# StanardScalerクラスをインポート
from sklearn.preprocessing import StandardScaler

# 各機械学習モデルをインポート
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
#評価のライブラリ
from sklearn.metrics import classification_report

def LogReg(X_train, X_test, y_train, y_test):

    #パイプライン作成
    pl = Pipeline([
       ('scl', StandardScaler()),
        ('lgr', LogisticRegression())
    ])

    #train with train data
    pl.fit(X_train, y_train)
    
    #Estimate test value
    pred = pl.predict(X_test)
    prob = pl.predict_proba(X_test)
    
    #Evaluation
    print(classification_report(y_test, pred))
    
    return pred, prob