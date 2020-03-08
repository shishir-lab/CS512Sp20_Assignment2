

import string 
import numpy as np
import torch
import torch.utils.data as data_utils

#%%

def read_train(filename):
    mapping = list(enumerate(string.ascii_lowercase))
    mapping = {i[1]: i[0] for i in mapping}

    file = open(filename, "r")
    file_data= file.read()
    file_data = file_data.split("\n")
    file.close()
    
    X, Y, tempX, tempY = [], [], [], []
    for col in file_data[:-1]:
        col = col.split(" ")
        tempY.append(mapping[col[1]])
        tempX.append(np.array(col[5:], dtype=float))
        if int(col[2]) == -1:
            X.append(np.array(tempX))
            Y.append(np.array(tempY, dtype=int))
            tempX.clear()
            tempY.clear()
        else:
            pass

    XY_zip = zip(X,Y)
    return list(XY_zip)
#%%

def read_tensor_padding(filename):
    mapping = list(enumerate(string.ascii_lowercase))
    mapping = {i[1]: i[0] for i in mapping}

    file = open(filename, "r")
    file_data= file.read()
    file_data = file_data.split("\n")
    file.close()
    
    X, Y, tempX, tempY = [], [], [], []
    for col in file_data[:-1]:
        col = col.split(" ")
        tempY.append(mapping[col[1]])
        tempX.append(np.array(col[5:], dtype=float))
        if int(col[2]) == -1:
            X.append(np.array(tempX))
            Y.append(np.array(tempY, dtype=int))
            tempX.clear()
            tempY.clear()
        else:
            pass
    
    max_chars = max([len(y) for y in Y])
    print(max_chars)
    XY_zip = zip(X,Y)
    X = []
    Y = []
    for x,y in XY_zip:
        nChars, nFeat = x.shape
        pad = np.zeros((max_chars - nChars, nFeat))
        x = np.row_stack((x, pad))
        one_hot = np.zeros((max_chars, 26))
        one_hot[np.arange(len(y)),y] = 1
        X.append(x)
        Y.append(one_hot)
    tensor_data = data_utils.TensorDataset(torch.tensor(X).double(), torch.tensor(Y).long())
    return tensor_data

#%%
train_data = read_tensor_padding('../data/train.txt')
#
#print(train_data[-1])
#%%
train_data[0]
