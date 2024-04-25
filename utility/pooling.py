import numpy as np
from util import im2col as i2c


class Pooling:
    def __init__(self, pool_h, pool_w, stride = 1, pad = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stide = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h)/ self.stride)
        out_w = int(1 + (W - self.pool_w)/ self.stride)
        # Expansion (1)

        col = i2c(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # Maximum value (2)
        out = np.max(col, axis=1)
        # Reshape (3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out