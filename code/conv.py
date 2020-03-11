import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Conv(nn.Module):
    """
    Convolution layer.
    """
    def __init__(self,kernel_size,in_channels=1,out_channels=1, stride=1, 
                       padding=0):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = Parameter(torch.empty(1, 1, kernel_size, kernel_size), requires_grad=True)
        self.K.data.normal_()
        self.kernel_size=kernel_size
        self.stride = stride
        self.padding = padding
        
    def init_params(self):
        """
        Initialize the layer parameters
        :return:
        """
    def load_params(self, K):
        if K.shape[0] == 1 and K.shape[1] == 1 and \
            K.shape[2] == self.kernel_size and K.shape[3] == self.kernel_size:
            self.K.data = K
        else:
            raise Exception("The Kernel size does not match. Excepted K of dimension {} x {}".format(self.kernel_size, self.kernel_size))

    def forward(self,X):
        
        # The following 2 lines are needed when padding is not zero (no need for zero padding)
        pad=(self.padding, self.padding, self.padding, self.padding)	
        X = nn.functional.pad(X, pad)
        # Unfolding rows and columns of X, so X and K could be multplied. unfold(dimension, size, step)
        X_all = X.unfold(2,self.kernel_size,self.stride).unfold(3,self.kernel_size,self.stride)
        #X_ij = X_all[0,0,:,:,:,:] this does not support multiple batches and/or multiple channels
        batch, channel, row=[], [], []
        for batches in X_all:
            for channels in batches:
                for rows in channels:
                    column = []
                    for columns in rows:
                        arr = torch.sum(torch.mul(columns,self.K))
                        column.append(arr)
                    row.append(column)
                channel.append(row)
            batch.append(channel)
        batch_tensor = torch.Tensor(batch)
        return batch_tensor
    
    # def backward(self):
        """
        Backward pass 
        (leave the function commented out,
        so that autograd will be used.
        No action needed here.)
        :return:
        """