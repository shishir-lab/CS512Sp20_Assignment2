from batch_custom_Conv import myConv
from read_data import read_tensor_padding
print("Loading Dataset...")
trainSet = read_tensor_padding('data/train.txt')
testSet = read_tensor_padding('data/train.txt')
print("Dataset Loaded ...")
words = trainSet[:][0]
labels = trainSet[:][1]
#%%
trainConv = trainSet[:][0]
batch_size, max_chars, dimX = trainConv.shape
all_images = trainConv.reshape(-1,1,16,8)
conv_layer = myConv(in_chan=1, out_chan=1, k_size=5, batch_size=2, stride=[1, 1], pad=True, b=False)
print(all_images[:2].shape)
out = conv_layer(all_images[:2])
print(out.shape)
#out = out.view(-1, 64)
out = out.view(-1, out.shape[2]*out.shape[3])
#out = out.reshape(batch_size, max_chars, -1)
print(out)
