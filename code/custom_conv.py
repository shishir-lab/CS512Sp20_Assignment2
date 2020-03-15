import torch
import torch.nn as nn
import math

#  torch Conv2d layer
class builtinConv(nn.Module):
    def __init__(self, in_chan, out_chan, k_size, str, pad, b):
        super(builtinConv, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, k_size, stride=str, padding=pad, bias=b).double()

    def forward(self, X):
        out = self.conv(X)
        return out


# custom convolution layer
class myConv(nn.Module):
    """
    myConvolution layer.
    """

    def __init__(self, in_chan, out_chan, k_size, stride, pad, b=False):
        super(myConv, self).__init__()
        self.in_channels = in_chan
        self.out_channels = out_chan
        self.kernel_size = k_size
        self.stride = stride
        self.str_row = stride[0]
        self.str_col = stride[1]
        self.padding = pad
        self.bias = b
        self.weight = nn.Parameter(torch.randn((1, 1, self.kernel_size, self.kernel_size), dtype=torch.float64))

    def init_params(self, X):
         pass

    def forward(self, X):
        batch_size = X.shape[0]
        if self.padding > 0:  # zero pad input if padding is a positive int
            m = nn.ZeroPad2d(self.padding)
            X = m(X)
        X_rows = X.shape[2]  # num rows of input
        X_cols = X.shape[3]  # num columns of input
        K_rows = self.weight.shape[2]
        K_cols = self.weight.shape[3]
        result_xdim = math.floor(((X_rows - (K_rows-1)-1) / self.str_row) + 1)  # Xdim of output
        result_ydim = math.floor(((X_cols - (K_cols-1)-1) / self.str_col) + 1)  # ydim of output
        result = torch.zeros([batch_size, self.out_channels, result_xdim, result_ydim], dtype=torch.float64)
        for i in range(0, result_xdim):
            for j in range(0, result_ydim):
                #  calculates a single activation for all samples in the batch using array slicing
                batch_conv_units = torch.sum(torch.sum(torch.mul(self.weight, torch.narrow(torch.narrow(X, 2, i*self.str_row, K_rows), 3, j*self.str_col, K_cols)), -1),-1)
                result[:, :, i, j] = batch_conv_units
        return result
