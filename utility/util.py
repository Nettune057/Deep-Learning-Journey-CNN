import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """Receives multiple images as input and converts them into a two-dimensional array (flattening).
    
    Parameters
    ----------
    input_data: Input data in the form of a 4-dimensional array (number of images, number of channels, height, width)
    filter_h: Height of filter
    filter_w: Width of filter
    stride: stride
    pad: padding
    
    Returns
    -------
    col: two-dimensional array
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(Contrary to im2col) It takes a two-dimensional array as input and converts it into a bundle of multiple images.
    
    Parameters
    ----------
    col: 2-dimensional array (input data)
    input_shape: Shape of the original image data (example: (10, 1, 28, 28))
    filter_h: Height of filter
    filter_w: Width of filter
    stride: stride
    pad: padding
    
    Returns
    -------
    img: converted images
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]