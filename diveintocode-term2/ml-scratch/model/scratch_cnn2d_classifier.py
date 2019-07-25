import sys
dir_str = "../../ml-scratch/utils"
if (dir_str not in sys.path):
    sys.path.append(dir_str)

import numpy as np
from collections import OrderedDict
from get_mini_batch import GetMiniBatch
from optimizer_2 import SGD, Momentum, Nesterov, AdaGrad, RMSprop, Adam
from activator_2 import Relu, Softmax
from layer import Conv2D, MaxPooling2D, Flatten, Affine

class Scratch2dCNNClassifier():
    def __init__(
        self, conv_param={'n_filters': 30, 'filter_size': 5, 'stride': 1, 'pad': 0},
        pool_param={'pool_size': 2},
        n_epochs=5, batch_size=100, optimizer='Adam',
        optimizer_param={'lr': 0.001},
        layer_nodes = {'hidden': 100, 'output': 10},
        weight_init_std=0.01,
        verbose=True
    ):
        self.conv_param = conv_param
        self.pool_param = pool_param
        self.layer_nodes = layer_nodes
        self.weight_init_std = weight_init_std
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose

        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov,
                               'adagrad': AdaGrad, 'rmsprop': RMSprop, 'adam': Adam}
        #**kwargsでdictで引数をまとめて受け取っている
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_loss_list =[]
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        #layerを生成
        self._gen_layers()
        #epoch数だけ学習
        for epoch in range(self.n_epochs):
            self._train()
            print("epoch: " + str(epoch))
            #verbose=Trueなら学習中のlossなど計算して表示
            if (self.verbose):
                self._calc_loss_acc()
                print("train_acc: " + str(self.train_acc_list[epoch]) + ", val_acc" + str(self.val_acc_list[epoch]))
                print("train loss: " + str(self.train_loss_list[epoch]) + ", val_loss" + str(self.val_loss_list[epoch]) )
        return self.train_loss_list, self.train_loss_list

    def predict(self, x):
        proba = self._propagate_forward(x)
        return np.argmax(proba, axis=1)

    def _gen_layers(self):
        """
        x_train: ndarray of shape(n_samples, n_channels, height, width)
        """
        #とりあえず画像サイズは正方形を想定し、input_size= heightとする
        self.n_train_samples, n_channels, input_size, _ = self.x_train.shape
        n_filters = self.conv_param['n_filters']
        filter_size = self.conv_param['filter_size']
        filter_stride = self.conv_param['stride']
        filter_pad = self.conv_param['pad']
        pool_size = self.pool_param['pool_size']

        conv_output_size = 1 + (input_size - filter_size + 2 * filter_pad) / filter_stride
        pool_output_size = int(n_filters * np.power(conv_output_size/ pool_size, 2))

        #initialize hyper parameters
        self.params ={}
        self.params['W1'] = self.weight_init_std * np.random.randn(n_filters, n_channels, filter_size, filter_size)
        self.params['b1'] = np.zeros(n_filters)
        self.params['W2'] = self.weight_init_std * np.random.randn(pool_output_size, self.layer_nodes['hidden'])
        self.params['b2'] = np.zeros(self.layer_nodes['hidden'])
        self.params['W3'] = self.weight_init_std * np.random.randn(self.layer_nodes['hidden'], self.layer_nodes['output'])
        self.params['b3'] =  np.zeros(self.layer_nodes['output'])

        #generate layers
        self.layers = OrderedDict()
        self.layers['Conv1'] = Conv2D(self.params['W1'], self.params['b1'], filter_stride, filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = MaxPooling2D(pool_h=pool_size, pool_w=pool_size, stride=pool_size)
        #ここにFlatten()挟む
        self.layers['Flatten1'] = Flatten()
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Last'] = Softmax()

        #gradients
        self.grads = {}

    def _train(self):
        mini_batch = GetMiniBatch(X=self.x_train, y=self.y_train, batch_size=self.batch_size, seed=0)

        for mini_x, mini_y in mini_batch:
            #forward
            z = self._propagate_forward(mini_x)
            #backward
            self._propagate_backward(z - mini_y)
            #gradient更新
            self.optimizer.update(self.params, self.grads)
        return

    def _loss(self, y_actual, pred_proba):
        return -(y_actual * np.log(pred_proba + 1e-7)).sum() / y_actual.shape[0]

    def _accuracy(self, y_actual, pred_proba):
        y_actual = np.argmax(y_actual, axis=1)
        pred = np.argmax(pred_proba, axis=1)
        acc = np.sum( y_actual == pred) / y_actual.shape[0]
        return acc

    def _propagate_forward(self, x):
        #forward
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def _propagate_backward(self, dout):
        #backward
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        #ここはupdate weightsとして切り出したほうがわかりやすいかも
        self.grads['W1'], self.grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        self.grads['W2'], self.grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        self.grads['W3'], self.grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        #返却値なし
        return
    def _calc_loss_acc(self):
        proba = self._propagate_forward(self.x_train)
        #loss計算
        loss = self._loss(self.y_train, proba)
        self.train_loss_list.append(loss)
        #accuracy計算
        train_acc = self._accuracy(self.y_train, proba)
        self.train_acc_list.append(train_acc)

        if((self.x_val is not None) & (self.y_val is not None)):
            proba = self._propagate_forward(self.x_val)
            #loss計算
            val_loss = self._loss(self.y_val, proba)
            self.val_loss_list.append(loss)
            #accuracy計算
            val_acc = self._accuracy(self.y_val, proba)
            self.val_acc_list.append(val_acc)
        #返却値なし
        return
