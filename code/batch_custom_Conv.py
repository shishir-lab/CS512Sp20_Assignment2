import torch
import torch.nn as nn
import math
import torch.optim as optim
torch.manual_seed(123)
torch.cuda.manual_seed(123)


class builtinConv(nn.Module):
    def __init__(self, in_chan, out_chan, k_size, str, pad, b):
        super(builtinConv, self).__init__()
        K = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float64).view(1,1,3,3)
        self.conv1 = nn.Conv2d(in_chan, out_chan, k_size, stride=str, padding=pad, bias=b).double()
        self.conv1.weight.data = K

    def forward(self, X):
        out = self.conv1(X)
        return out



#custom convolutional net from scratch
class myConv(nn.Module):
    """
    myConvolution layer.
    """

    def __init__(self, in_chan, out_chan, k_size, batch_size, stride, pad, b):
        super(myConv, self).__init__()
        self.in_channels = in_chan
        self.out_channels = out_chan
        self.kernel_size = k_size
        self.str_row = stride[0]
        self.str_col = stride[1]
        self.padding = pad
        self.bias = b
        self.filter_weights = nn.Parameter(torch.randn((1, 1, self.kernel_size, self.kernel_size), dtype=torch.float64))
        #self.filter_weights = nn.Parameter(torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float64).view(1, 1, 3, 3))
        self.batch_size = batch_size

    def init_params(self, X):
         pass

    def forward(self, X):
        if self.padding == True:
            m = nn.ZeroPad2d(1)
            X = m(X)
        X_rows = X.shape[2]
        X_cols = X.shape[3]
        K_rows = self.filter_weights.shape[2]
        K_cols = self.filter_weights.shape[3]
        result_xdim = math.floor(((X_rows - (K_rows-1)-1) / self.str_row) + 1)
        result_ydim = math.floor(((X_cols - (K_cols-1)-1) / self.str_col) + 1)
        result = torch.zeros([self.batch_size, self.out_channels, result_xdim, result_ydim], dtype=torch.float64)
        for i in range(0, result_xdim):
            for j in range(0, result_ydim):
                #calculates a single activation for all samples in the batch by using torch.narrow()
                batch_conv_units = torch.sum(torch.sum(torch.mul(self.filter_weights, torch.narrow(torch.narrow(X, 2, i*self.str_row, K_rows), 3, j*self.str_col, K_cols)), -1),-1)
                print(batch_conv_units)
                result[:, :, i, j] = batch_conv_units
        print(result)
        return result


if __name__ == '__main__':
    # initialize layer hyperparams
    batch_size=10
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    stride=[2,2]
    stride2=(2,2)
    bias = False
    padding = True
    pad = 1

    # input tensor
    X_batch = batch_size*[[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]]
    X_batch = torch.tensor(X_batch, dtype=torch.float64).view(batch_size, 1, 5, 5)


    # instantiate from  my_conv and builtin_conv to compare grads
    my_conv = myConv(in_channels, out_channels, kernel_size, batch_size, stride, padding, bias)
    builtin_conv = builtinConv(in_channels, out_channels, kernel_size, stride2, pad, bias)

    # define loss function and optimizer
    loss_fn = torch.nn.MSELoss(reduction='sum')
    #optimizer1 = optim.SGD(my_conv.parameters(), lr=0.005, momentum=0.9)
    #optimizer2 = optim.SGD(builtin_conv.parameters(), lr=0.005, momentum=0.9)

    optimizer1 = optim.Adam(my_conv.parameters(), lr=0.01)
    optimizer2 = optim.Adam(builtin_conv.parameters(), lr=0.01)

    # get output and calculate loss
    output1 = my_conv(X_batch)
    output2 = builtin_conv(X_batch)
    print(output1, 'my conv output') #10*3*3
    print(output2, ' builtin output')

    #generate random label
    rand_output = torch.randn(output1.size(), dtype=torch.float64)

    #calculate loss1 and loss2 and do backprop
    loss1 = loss_fn(output1, rand_output)
    #loss1= F.hinge_embedding_loss(output1, rand_output,reduction='mean')

    loss2 = loss_fn(output2, rand_output)
    #loss2 = F.hinge_embedding_loss(output2, rand_output,reduction='mean')

    loss1.backward(retain_graph=True)
    loss2.backward(retain_graph=True)

    for i in range(1000): #train both networks for 300 iterations, the grad and loss values for both are the same...
        output1 = my_conv(X_batch)
        output2 = builtin_conv(X_batch)

        my_conv.zero_grad()
        builtin_conv.zero_grad()
        loss1 = loss_fn(output1, rand_output)
        loss2 = loss_fn(output2, rand_output)
        loss1.backward()
        loss2.backward()
        print('loss1 value in forward pass', loss1)
        print('loss2 value in forward pass', loss2)


        optimizer1.step()
        optimizer2.step()
        print(my_conv.filter_weights.grad)
        print(builtin_conv.conv1.weight.grad)

