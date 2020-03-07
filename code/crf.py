import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
#%%
def crf_decode(W, T, word, dimX, dimY):
    """
    Dynamic Programming implementation of CRF Decoding with pytorch.
    """
    char_count = torch.nonzero(word.sum(axis=1)).size(0)
    # Construct lookup
    lookup = torch.zeros((char_count, dimY))
    for i in range(1,char_count): # dot product only upto second last element
        first_term = W.t().matmul(word[i-1]) # $W_{yi-1} . X_{i-1}$
        for y1 in range(0, dimY): # for each next_label
            second_term = T[:,y1] # $T_{yi-1, yi}$
            sum_term = first_term + second_term + lookup[i-1]
            lookup[i,y1] = torch.max(sum_term) # get best score by all possible current_label
    
    # BackTrack to get the solution
    previousAns = [None]
    score = 0
    for i in reversed(range(0,char_count)): # go from last to first
        first_term = W.t().matmul(word[i])
        if previousAns[-1] is None: #last label does not have next_label
            score = torch.max(first_term + lookup[i]) # lookup contains best score 
            print(score) # Score to be reported
            ans = torch.argmax(first_term + lookup[i])
            previousAns[0] = ans
        else:
            second_term = T[:, previousAns[-1]]
            ans = torch.argmax(first_term+second_term+lookup[i])
            previousAns.append(ans)
            
    previousAns.reverse()
    return previousAns
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
class CRFFunc(torch.autograd.function.Function):

  @staticmethod
  def forward(ctx, W, T, train, dimX, dimY):
    ctx.dimX = dimX
    ctx.dimY = dimY
    ctx.save_for_backward(W, T, train)
    # compute prediction
    batch_size = train.size(0)
    max_chars = train.size(1)
    predictions = torch.zeros(batch_size, max_chars, dimY)
    for i,word in enumerate(train):
      preds = crf_decode(W,T,word,dimX,dimY)
      one_hot_preds = torch.zeros(max_chars, dimY, dtype=torch.long)
      one_hot_preds[torch.arange(len(preds)), preds] = 1
      predictions[i] = one_hot_preds
    return predictions
  
  @staticmethod
  def backward(ctx, grad_output):
    dimX = ctx.dimX
    dimY = ctx.dimY
    W, T, train = ctx.saved_tensors
    grad_W = grad_T = None
    # calculate the gradient for the loss function
    return grad_W, grad_T, None, None, None

#%%


class CRF(nn.Module):

    def __init__(self, input_dim, embed_dim, conv_layers, num_labels, batch_size):
        """
        Linear chain CRF as in Assignment 2
        @param input_dim: Number of features in raw input (default: 128)
        @param embed_dim: #TODO decode (default: 64)
        @param conv_layers: #TODO decode (default: [[1, 64, 128]])
        @param num_labels: Number of labels in raw input (default: 26)
        @param batch_size: Number of training examples in a batch (default: 64)
        """
        super(CRF, self).__init__()
        self.input_dim = input_dim 
        self.embed_dim = embed_dim
        self.conv_layers = conv_layers
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        # Replicating old CRF on pytorch
        self.W = Parameter(torch.empty(self.input_dim, self.num_labels), requires_grad=True)
        self.T = Parameter(torch.empty(self.num_labels, self.num_labels), requires_grad=True)
        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]
        self.init_params()

    def init_params(self):
        """
        Initialize trainable parameters of CRF here
        """        
        for param in self.parameters():
          param.data.zero_()
          
    def load_params(self, W, T):
        self.W.data = W
        self.T.data = T

    def forward(self, X):
        """
        Implement the objective of CRF here.
        The input (features) to the CRF module should be convolution features.
        """
        # features = self.get_conv_feats(X) # TODO Implement it
        prediction = CRFFunc.apply(self.W, self.T, X, self.input_dim, self.num_labels)
        return prediction

    def loss(self, X, labels):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        features = self.get_conv_feats(X)
        loss = None # TODO calculate loss
        return loss


    def get_conv_features(self, X):
        """
        Generate convolution features for a given word
        """
        convfeatures = None # TODO implement
        return convfeatures
#%%
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
test_crf_forward()
#%%
