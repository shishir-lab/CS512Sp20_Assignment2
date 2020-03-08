import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from crf_utils import crf_decode, crf_gradient, crf_logloss
torch.set_default_dtype(torch.float64)
#%%
class CRFLoss(torch.autograd.function.Function):

  @staticmethod
  def forward(ctx, W, T, words, labels, C, dimX, dimY):
    ctx.dimX = dimX
    ctx.dimY = dimY
    ctx.C = C
    ctx.save_for_backward(W, T, words, labels)
    log_crf = 0
    for i in range(words.shape[0]):
        log_crf += crf_logloss(W, T, words[i], labels[i], dimX, dimY)
    
    log_crf = log_crf / words.shape[0]
    # print(log_crf)
    f = (-C *log_crf)  + 0.5 * torch.sum(W*W) + 0.5 * torch.sum(T*T)
    return f
    
  @staticmethod
  def backward(ctx, grad_output):
    print("Called backward function...")
    print(grad_output)
    dimX = ctx.dimX
    dimY = ctx.dimY
    C = ctx.C
    W, T, words, labels = ctx.saved_tensors
    avg_grad_W = torch.zeros(dimY, dimX)
    avg_grad_T = torch.zeros(dimY, dimY)
    
    for i in range(words.shape[0]):
        gradW, gradT = crf_gradient(W, T, words[i], labels[i], dimX, dimY)
        avg_grad_W += gradW
        avg_grad_T += gradT
    avg_grad_W = avg_grad_W / words.shape[0]
    avg_grad_T = avg_grad_T / words.shape[0]
        
    grad_W = -C * avg_grad_W.T + W
    grad_T = -C * avg_grad_T + T
    grad_W = grad_W * (grad_output)
    grad_T = grad_T * (grad_output)
    # calculate the gradient for the loss function
    return grad_W, grad_T, None, None, None, None, None

#%%


class CRF(nn.Module):

    def __init__(self, input_dim, embed_dim, conv_layers, num_labels, batch_size, C=1000):
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
        # TODO handle zero-padding
        self.C = C
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
        dimX = self.input_dim
        dimY = self.num_labels
        batch_size = X.size(0)
        max_chars = X.size(1)
        predictions = torch.zeros(batch_size, max_chars, dimY)
        for i,word in enumerate(X):
          preds = crf_decode(self.W,self.T,word,dimX,dimY)
          one_hot_preds = torch.zeros(max_chars, dimY, dtype=torch.long)
          one_hot_preds[torch.arange(len(preds)), preds] = 1
          predictions[i] = one_hot_preds
        return predictions

    def loss(self, words, labels):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        # features = self.get_conv_feats(X)
        dimX = self.input_dim
        dimY = self.num_labels
        # Commented code for utilizing pytorch autograd and checking grads
        # W = self.W
        # T = self.T
        # C = self.C

        # log_crf = 0
        # for i in range(words.shape[0]):
        #     log_crf += crf_logloss(W, T, words[i], labels[i], dimX, dimY)
        
        # log_crf = log_crf / words.shape[0]
        # # print(log_crf)
        # f = (-C *log_crf)  + 0.5 * torch.sum(W*W) + 0.5 * torch.sum(T*T)
        # return f

        return CRFLoss.apply(self.W, self.T, words, labels, self.C, dimX, dimY)
    


    def get_conv_features(self, X):
        """
        Generate convolution features for a given word
        """
        convfeatures = None # TODO implement
        return convfeatures
#%%

    