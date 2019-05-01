import numpy as np

def train_test_split(X, y, train_size=0.8):
    """
    学習用データを分割する。
    
    Parameters
    ----------
    X : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    y : 次の形のndarray, shape (n_samples, )
      正解値
    train_size : float (0<train_size<1)
      何割をtrainとするか指定

    Returns
    ----------
    X_train : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    X_test : 次の形のndarray, shape (n_samples, n_features)
      検証データ
    y_train : 次の形のndarray, shape (n_samples, )
      学習データの正解値
    y_test : 次の形のndarray, shape (n_samples, )
      検証データの正解値
    """

    #n of rows ( number of samples)
    n_samples = X.shape[0]
    #n of columns ( number of features)
    n_features = X.shape[1]
    
    #generate ndarray from 0 to n_samples (index)
    index_rnd = np.array(range(n_samples))
    #shuffle index
    np.random.shuffle(index_rnd)
    
    #calculate number of train samples 
    n_train_samples = int(n_samples * train_size)
    
    #train samples
    X_train = X[index_rnd[:n_train_samples]]
    y_train = y[index_rnd[:n_train_samples]]
    
    #test samples
    X_test = X[index_rnd[n_train_samples:]]
    y_test = y[index_rnd[n_train_samples:]]
    
    pass

    return X_train, X_test, y_train, y_test
