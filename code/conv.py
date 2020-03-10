import torch
import torch.nn as nn


class Conv(nn.Module):
    """
    Convolution layer.
    """
    def __init__(self,K,K_size,stride,padding):
        super(Conv, self).__init__()
        self.K = K
        self.K_size=K_size
        self.stride = stride
        self.padding = padding
        
    def init_params(self):
        """
        Initialize the layer parameters
        :return:
        """

    def forward(self,X):
        
        # The following 2 lines are needed when padding is not zero (no need for zero padding)
        pad=(self.padding, self.padding, self.padding, self.padding)	
        X = nn.functional.pad(X, pad)
        # Unfolding rows and columns of X, so X and K could be multplied. unfold(dimension, size, step)
        X_all = X.unfold(2,self.K_size,self.stride).unfold(3,self.K_size,self.stride)
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