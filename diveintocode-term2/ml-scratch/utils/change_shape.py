import numpy as np
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
    out_h = (H + 2 * pad - filter_h)//stride + 1 #/でfloat, //でintegerを返す
    out_w = (W + 2 * pad - filter_w)//stride + 1
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
    out_h = (H + 2 * pad - filter_h)//stride + 1
    out_w = (W + 2 * pad - filter_w)//stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    col = col.transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad +  stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
