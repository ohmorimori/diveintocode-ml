import numpy as np
def get_output_size(n_features, filter_length, stride=1, pad=0):
    return int(1 + (n_features + 2 * pad - filter_length) / stride)#/でfloat, //でintegerを返す ここではint()を使用

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Paramaters
    ----------------
    input_data: ndarray of shape(n_data, channel, height, width)
    filter_h: int
        filter height
    filter_w: int
        filter width
    stride: int
    pad: int

    Returns
    -----------------
    col: 2D array
    """

    N, C, H, W = input_data.shape
    out_h = get_output_size(n_features=H, filter_length=filter_h, stride=stride, pad=pad)
    out_w = get_output_size(n_features=W, filter_length=filter_w, stride=stride, pad=pad)
    #padding
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    #start = time.time()
    for y in range(filter_h):
        y_max = y + stride * out_h
        #print("y", y, "y_max", y_max)
        for x in range(filter_w):
            x_max = x + stride * out_w
            #print("x", x, "x_max", x_max)
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            #print("y, x, y_max, x_max: ", y, x, y_max, x_max)
            #print("col:", col)
    #lap = time.time()
    #print("time: ", lap - start, " sec")
    col = col.transpose(0, 4, 5, 1, 2, 3)
    #print("col:", col)
    col = col.reshape(N * out_h * out_w, -1)
    #print("col:", col)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    -----------------
    col: data to convert
    input_shape:
        i.e. shape(10, 1, 28, 28)
    filter_h: int
        filter_height
    filter_w: int
        filter_width
    stride: int
    pad: int

    Returns
    -----------------
    ndarray of shape input_shape
    """
    N, C, H, W = input_shape
    out_h = get_output_size(n_features=H, filter_length=filter_h, stride=stride, pad=pad)
    out_w = get_output_size(n_features=W, filter_length=filter_w, stride=stride, pad=pad)

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    col = col.transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad +  stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def im2col_1d(input_data, filter_size, stride=1, pad=0):
    """
    Paramaters
    ----------------
    input_data: ndarray of shape(n_data, channel, n_features)
    filter_size: int
    stride: int
    pad: int

    Returns
    -----------------
    col: 1D array
    """

    N, C, L = input_data.shape#
    out_size = get_output_size(n_features=L, filter_length=filter_size, stride=stride, pad=pad)

    #padding
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_size, out_size))

    for f in range(filter_size):
        f_max = f + stride * out_size
        col[:, :, f, :] = img[:, :, f:f_max:stride]

    col = col.transpose(0, 3, 1, 2)
    col = col.reshape(N * out_size, -1)
    return col

def col2im_1d(col, input_shape, filter_size, stride=1, pad=0):
    """
    Parameters
    -----------------
    col: data to convert
    input_shape:
        i.e. shape(10, 1, 784)
    filter_size: int
    stride: int
    pad: int

    Returns
    -----------------
    ndarray of shape input_shape
    """
    N, C, L = input_shape#shape(n_data, channel, n_features)
    out_size = get_output_size(n_features=L, filter_length=filter_size, stride=stride, pad=pad)

    col = col.reshape(N, out_size, C, filter_size).transpose(0, 2, 3, 1)
    img = np.zeros((N, C, L + 2 * pad +  stride - 1))
    for f in range(filter_size):
        f_max = f + stride * out_size
        img[:, :, f:f_max:stride] += col[:, :, f, :]

    return img[:, :, pad:L + pad]
