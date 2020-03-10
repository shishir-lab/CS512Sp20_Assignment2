import torch
import torch.nn as nn
import math
import torch.optim as optim

#builtin convolutional net Conv2d
class builtinConv(nn.Module):
    def __init__(self, in_chan, out_chan, k_size, str, pad, b):
        super(builtinConv, self).__init__()
        #K = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float).view(1, 1, 3, 3)
        self.conv1 = nn.Conv2d(in_chan, out_chan, k_size, stride=str, padding=pad, bias=b)
        #self.conv1.weight.data = K

    def forward(self, X):
        return self.conv1(X)

#custom convolutional net from scratch
class myConv(nn.Module):
    """
    myConvolution layer.
    """

    def __init__(self, in_chan, out_chan, k_size, str, pad=False,
                 b=False):
        super(myConv, self).__init__()
        self.in_channels = in_chan
        self.out_channels = out_chan
        self.kernel_size = k_size
        self.stride = str
        self.padding = pad
        self.bias = b
        self.padding = pad
        #self.filter_weights = nn.Parameter(torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float).view(3, 3))
        self.filter_weights = nn.Parameter(torch.randn(kernel_size, kernel_size)) #uncomment this line for random initialization
        self.filter_weights.requires_grad = True

    def init_params(self, X):
        X_rows = X.shape[0]
        X_cols = X.shape[1]
        K_rows = self.filter_weights.shape[0]
        K_cols = self.filter_weights.shape[1]

        result = torch.zeros(
            [math.floor((X_rows - K_rows + stride) / stride), math.floor((X_cols - K_cols + stride) / stride)],
            dtype=torch.float32)
        for k, i in enumerate(range(0, X_rows - K_rows + 1, stride)):
            for p, j in enumerate(range(0, X_cols - K_cols + 1, stride)):
                conv_unit = torch.sum(torch.mul(self.filter_weights, X[i:i + K_rows, j:j + K_cols]))
                result[k, p] = conv_unit
        # print(result)
        print(result.requires_grad)
        # features = result.clone().detach().requires_grad_(True)
        return result

    def forward(self, X):
        if self.padding == True:  # input is zero padded if padding set to True
            m = nn.ZeroPad2d(1)
            X = m(X)
        output = self.init_params(X)
        return output
        # def backward(self):
        """
        Backward pass 
        (leave the function commented out,
        so that autograd will be used.
        No action needed here.)
        :return:
        """


def single_forward_3a(X):
    net = myConv(1, 1, 3, 1, False, False)
    output = net(X)
    print('solution to 3a with no zero padding and stride=1:', output)


if __name__ == '__main__':
    # initialize layer hyperparams
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    stride = 1
    bias = False
    padding = False
    pad = 0

    # input tensor
    X = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.float).view(5, 5)

    # instantiate from  my_conv and builtin_conv to compare grads
    my_conv = myConv(in_channels, out_channels, kernel_size, stride, padding, bias)
    builtin_conv = builtinConv(in_channels, out_channels, kernel_size, stride, pad, bias)

    # define loss function and optimizer
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer1 = optim.SGD(my_conv.parameters(), lr=0.01, momentum=0.9)
    optimizer2 = optim.SGD(builtin_conv.parameters(), lr=0.01, momentum=0.9)

    # get output of layer my_conv and calculate loss
    output1 = my_conv(X)
    output2 = builtin_conv(X.view(1, 1, 5, 5))
    #generate random label
    rand_output = torch.randn(output1.size())
    #calculate loss1 and loss2 and do backprop
    loss1 = loss_fn(output1, rand_output)
    loss1.backward(retain_graph=True)
    loss2 = loss_fn(output2, rand_output.view(output2.size()))
    loss2.backward(retain_graph=True)

    for i in range(300): #train both networks for 300 iterations, the grad and loss values for both are the same...
        output1 = my_conv(X)
        output2 = builtin_conv(X.view(1, 1, 5, 5))
        my_conv.zero_grad()
        builtin_conv.zero_grad()
        loss1 = loss_fn(output1, rand_output)
        loss2 = loss_fn(output2, rand_output.view(output2.size()))
        loss1.backward()
        loss2.backward()
        print('loss1 value in forward pass', loss1)
        print('loss2 value in forward pass', loss2)

        optimizer1.step()
        optimizer2.step()
        print(my_conv.filter_weights.grad)
        print(builtin_conv.conv1.weight.grad)

    single_forward_3a(X) #gives solution to question 3a

    # to be done by using data_loader and training on batches of data
    #def train_custom_conv_features():
         #return conv_features