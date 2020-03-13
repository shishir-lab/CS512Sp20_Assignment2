import torch
from batch_custom_Conv import myConv , builtinConv

kernel_weights = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float64).view(1, 1, 3, 3)
X = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.float64).view(1, 1, 5, 5)


conv_layer = myConv(in_chan=1, out_chan=1, k_size=3, stride=[1, 1], pad=1, b=False)
conv_layer.weight.data = kernel_weights
output = conv_layer(X)
print('\nOutput of the custom convolution with zero padding and unit stride : \n',output)

conv_layer2 = builtinConv(in_chan=1, out_chan=1, k_size=3, str=(1,1), pad=1, b=False)
conv_layer2.conv1.weight.data = kernel_weights
output2 = conv_layer2(X)
print('\nOutput of the nn.conv2d with zero padding and unit stride : \n', output2)