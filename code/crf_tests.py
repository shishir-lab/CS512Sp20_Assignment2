# -*- coding: utf-8 -*-
#%%
import numpy as np
import torch
from crf_utils import crf_decode, crf_logloss, crf_gradient, compare
from crf import CRF, CRFLoss
#%%

# Testing decoder implementation
def test_decoder():
    X = np.loadtxt('../data/decode_input.txt')
    X = torch.tensor(X)
    x = X[:100*128].reshape(-1,100,128)
    w = X[100*128:-26*26].reshape(26, 128).t()
    t = X[-26*26:].reshape(26,26).t()
    print(x.shape, w.shape, t.shape)
    batch_size = x.size(0)
    max_chars = x.size(1)
    predictions = torch.zeros(batch_size, max_chars, 26)
    for i,word in enumerate(x):
      preds = crf_decode(w,t,word,128,26)
      one_hot_preds = torch.zeros(max_chars, 26, dtype=torch.long)
      one_hot_preds[torch.arange(len(preds)), preds] = 1
      predictions[i] = one_hot_preds
    print(torch.argmax(predictions[0], axis=1) + 1)
test_decoder()

#%%
# Testing crf forward
def test_crf_forward():
    X = np.loadtxt('../data/decode_input.txt')
    X = torch.tensor(X)
    x = X[:100*128].reshape(-1,100,128)
    w = X[100*128:-26*26].reshape(26, 128).t()
    t = X[-26*26:].reshape(26,26).t()
    print(x.shape, w.shape, t.shape)
    crf = CRF(conv_layers=None, C=1)
    crf.load_params(w, t)
    predictions = crf(x)
    for pred in predictions:
        print(torch.argmax(pred, axis=1)+1)
    return crf
mycrf = test_crf_forward()
#%%
# Testing crf log loss implementation
from read_data import read_train
import time
def test_crf_logloss():
    train_data = read_train("../data/train.txt")
    train = np.array(train_data)
    word_list = train[:,0]
    label_list = train[:,1]
    print("word_list shape :", word_list.shape)
    print("label_list shape :", label_list.shape)
    print("word shape:", word_list[3].shape)
    X = np.loadtxt("../data/model.txt")
    X = torch.tensor(X)
    w = X[:-26*26].reshape(26, 128).t()
    t = X[-26*26:].reshape(26,26).t()
    start = time.time()
    loss = 0
    for i in range(len(word_list)):
        loss += crf_logloss(w, t, torch.tensor(word_list[i]), 
                            torch.tensor(label_list[i], dtype=torch.long), 128, 26)
    end = time.time()
    print(loss/len(word_list))
    print("Time: ", end-start)

test_crf_logloss()
#%%
from crfmodel import CRFModel
def test_check_grad():
    DIMX=5
    DIMY=3
    n_words = 5
    n_chars = 2
    np.random.seed(3)
    word_list = np.random.randint(10,size=DIMX*n_chars*n_words).reshape(n_words,n_chars,DIMX).tolist()
    label_list = np.random.choice(range(1,DIMY+1),size=n_chars*n_words).reshape(n_words,n_chars).tolist()
    x = np.zeros((DIMX*DIMY)+DIMY*DIMY)
#    x= np.random.uniform(size=(DIMX*DIMY)+(DIMY*DIMY))
    model = CRFModel(DIMX, DIMY)
    model.load_X(x)
    W1, T1 = model._W, model._T.T
    print("W = ",W1.shape)
    print("T = ",T1.shape)
    train = np.zeros((n_words, 2), dtype='object')
    for i in range(n_words):
        tempX = []
        for word in word_list[i]:
            tempX.append(np.array(word, dtype=float))
        train[i][0] = np.array(tempX)
        train[i][1] = np.array(label_list[i], dtype=int)

    
        
    # print("CRF = ",log_crf_wrapper(x,train, DIMX, DIMY))
    return crf_gradient(torch.tensor(W1), torch.tensor(T1.T),
                         torch.tensor(train[0][0]), torch.tensor(train[0][1],dtype=torch.long)-1, DIMX, DIMY)
    # g = grad_crf_wrapper(x, train, DIMX, DIMY)
    # print("g = {}".format(g))
    # score = opt.check_grad(log_crf_wrapper, grad_crf_wrapper, x, *[train, DIMX, DIMY])
    # print("Score = ",score)
    # assert score < 1.0e-4
g = test_check_grad()
print(g)

#%%
def check_gradients():
    DIMX=20
    DIMY=4
    n_words = 10
    n_chars = 3
    np.random.seed(3)
    word_list = np.random.randint(10,size=DIMX*n_chars*n_words).reshape(n_words,n_chars,DIMX).tolist()
    label_list = np.random.choice(range(0,DIMY),size=n_chars*n_words).reshape(n_words,n_chars).tolist()
    words = torch.tensor(word_list, dtype=torch.double)
    labels = torch.tensor(label_list)
    W = torch.zeros(DIMX, DIMY, requires_grad=True).double()
    T = torch.zeros(DIMY, DIMY, requires_grad=True).double()
    C = 1
    print(words.shape, labels.shape, W.shape, T.shape)
    loss = CRFLoss.apply
    res = torch.autograd.gradcheck(loss, (W, T, words, labels, C, DIMX, DIMY), raise_exception=True)
    print(res)
    
check_gradients()
#%%
def check_gradients_module():
    DIMX=20
    DIMY=4
    n_words = 1
    n_chars = 3
    np.random.seed(3)
    word_list = np.random.randint(2,size=DIMX*n_chars*n_words).reshape(n_words,n_chars,DIMX).tolist()
    label_list = np.random.choice(range(0,DIMY),size=n_chars*n_words).reshape(n_words,n_chars).tolist()
    words = torch.tensor(word_list, dtype=torch.double)
    labels = torch.zeros(n_words, n_chars, DIMY)
    for i,label in enumerate(label_list):
        labels[i,torch.arange(n_chars),label] = 1
        
    # W = torch.zeros(DIMX, DIMY, requires_grad=True).double()
    # T = torch.zeros(DIMY, DIMY, requires_grad=True).double()
    # C = 1
    # print(words.shape, labels.shape, W.shape, T.shape)
    # loss = CRFLoss.apply
    # res = torch.autograd.gradcheck(loss, (W, T, words, labels, C, DIMX, DIMY), raise_exception=True)
    # print(res)
    print(words)
    print(labels)
    
    mycrf = CRF(input_dim=(5,4), num_labels=DIMY, conv_layers=[[3,1,1]], C=1)
    # W = torch.zeros(DIMX, DIMY, requires_grad=True).double()
    # T = torch.zeros(DIMY, DIMY, requires_grad=True).double()
    # C = 1
    loss = mycrf.loss(words, labels)
    loss.backward()
    # model = nn.Sequential(
    #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1, bias=False),
    #     mycrf
    #     )
    
    # res = torch.autograd.gradcheck(loss, (words, labels), raise_exception=True)
    print(mycrf.W.grad)
    print(mycrf.T.grad)
    print(mycrf.conv_layers[0].weight.grad)
    
check_gradients_module()

#%%
from crf import CRF, CRFLoss
from read_data import read_tensor_padding
import torch.optim as optim
print("Loading Dataset...")
trainSet = read_tensor_padding('../data/train.txt')
testSet = read_tensor_padding('../data/train.txt')
print("Dataset Loaded ...")

def test_optimize():
    words = trainSet[:][0]
    labels = trainSet[:][1]
    # mycrf = CRF(input_dim=(16,8), conv_layers=[[5,2,(2,1)]], num_labels=26, C=100)
    # mycrf = CRF(input_dim=(16,8), conv_layers=[], num_labels=26, C=100)

    # Setup the optimizer
    opt = optim.LBFGS(mycrf.parameters(), max_iter=100)
    start = time.time()
    disp = 0
    def closure():
        opt.zero_grad() # clear the gradients
        tr_loss = mycrf.loss(words, labels) # Obtain the loss for the optimizer to minimize
        tr_loss.backward() # Run backward pass and accumulate gradients
        if disp % 10 == 0:
            print(mycrf.conv_layers[0].weight.grad)
            print(tr_loss)
        return tr_loss
    
    opt.step(closure)
    end = time.time()
    print(end-start)
    with torch.no_grad():
        loss = mycrf.loss(words, labels)
        print(loss)
        preds = mycrf(testSet[:][0])
        compare(testSet[:][0], testSet[:][1], preds)
    return mycrf
mycrf = test_optimize()
#%%
# Testing 2D Conv networks
# torch.manual_seed(123)
# torch.cuda.manual_seed(123)
# np.random.seed(123)
# torch.backends.cudnn.enabled=False
# torch.backends.cudnn.deterministic=True
#%%
# import torch.nn as nn
# from read_data import read_tensor_padding
# print("Loading Dataset...")
# trainSet = read_tensor_padding('../data/train.txt')
# testSet = read_tensor_padding('../data/train.txt')
# print("Dataset Loaded ...")
# words = trainSet[:][0]
# labels = trainSet[:][1]
# #%%
# testcrf = CRF(input_dim=(16,8), conv_layers=[[5,2,(1,1)]], num_labels=26, C=100)
# # Setup the optimizer
# for param in testcrf.parameters():
#     print(param.shape)
# #%%
# testcrf.load_params(backupW.clone(), backupT.clone())
# #%%
# preds = testcrf(testSet[:][0])
# compare(testSet[:][0], testSet[:][1], preds)
# #%%
# opt = optim.LBFGS(testcrf.parameters(), max_iter=5)
# start = time.time()
# disp = 0
# def closure():
#     opt.zero_grad() # clear the gradients
#     tr_loss = testcrf.loss(words, labels) # Obtain the loss for the optimizer to minimize
#     tr_loss.backward() # Run backward pass and accumulate gradients
#     if disp % 10 == 0:
#         print(tr_loss)
#         print(testcrf.conv_layers[0].weight.grad)
#     return tr_loss

# opt.step(closure)
# end = time.time()
# print(end-start)
# with torch.no_grad():
#     loss = testcrf.loss(words, labels)
#     print(loss)
#     preds = testcrf(testSet[:][0])
#     compare(testSet[:][0], testSet[:][1], preds)
# # trainConv = trainSet[:][0]
# # batch_size, max_chars, dimX = trainConv.shape
# # all_images = trainConv.reshape(-1,1,16,8)
# # conv_layer = nn.Conv2d(in_channels=1, out_channels=1, groups=1,
# #                        kernel_size=5, stride=(2,1), padding=2, bias=False)
# # out = conv_layer(all_images[:2])
# # out = out.view(-1, 64)
# # # out = out.reshape(batch_size, max_chars, -1)
# # print(out)
# # print(out.shape)





    