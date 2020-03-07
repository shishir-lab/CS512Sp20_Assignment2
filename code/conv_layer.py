from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Q3 test case
X = np.reshape([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], (5, 5))
K = np.reshape([1, 0, 1, 0, 1, 0, 1, 0, 1], (3, 3))
zero_padding = False
X_rows = X.shape[0]
X_cols = X.shape[1]
K_rows = K.shape[0]
K_cols = K.shape[1]

def convolution(X, K, stride, zero_padding):
    X_rows = X.shape[0]
    X_cols=X.shape[1]
    K_rows = K.shape[0]
    K_cols = K.shape[1]

    output=[]
    for i in range(X_rows-K_rows+1):
        for j in range(X_cols-K_cols+1):
            conv_unit = np.sum(np.multiply(K, X[i:i+K_rows, j:j+K_cols]))
            output.append(conv_unit)
    print(output)
    output_matrix = np.reshape(output, (X_rows-K_rows+1, X_cols-K_cols+1))
    return output_matrix

if __name__=='__main__':
    print(convolution(X,K,1,False))
    input = torch.tensor(np.reshape(X, (1,1,X_rows,X_cols)))
    weights = torch.tensor(np.reshape(K, (1,1,3,3)))
    output = F.conv2d(input, weights)
    print(output)