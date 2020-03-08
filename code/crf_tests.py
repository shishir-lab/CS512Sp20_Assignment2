# -*- coding: utf-8 -*-
#%%
import numpy as np
import torch
from crf_utils import crf_decode, crf_logloss, crf_gradient
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
    crf = CRF(128, 64, None, 26, 1)
    crf.load_params(w, t)
    predictions = crf(x)
    for pred in predictions:
        print(torch.argmax(pred, axis=1)+1)
    return crf
crf = test_crf_forward()
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
from crf import CRF, CRFLoss
from read_data import read_tensor_padding
# def test_crf_impl():
trainSet = read_tensor_padding('../data/train.txt')
words = trainSet[:][0]
labels = trainSet[:][1]
X = np.loadtxt("../data/model.txt")
X = torch.tensor(X)
w = X[:-26*26].reshape(26, 128).t()
t = X[-26*26:].reshape(26,26).t()
mycrf = CRF(128, 64, None, 26, 1, C=1000)
mycrf.load_params(w,t)
start = time.time()
loss = mycrf.loss(words, labels)
print(loss)
#%%
start = time.time()
loss.backward()
end = time.time()
print(end-start)


    