import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from crf_utils import crf_decode, crf_gradient, crf_logloss
from Q3a import myConv
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
    # print("Called backward function...")
    # print(grad_output)
    dimX = ctx.dimX
    dimY = ctx.dimY
    C = ctx.C
    W, T, words, labels = ctx.saved_tensors
    
    batch_size, max_chars, embedDim = words.shape
    avg_grad_W = torch.zeros(dimY, dimX)
    avg_grad_T = torch.zeros(dimY, dimY)
    avg_grad_word = torch.zeros(batch_size, max_chars, embedDim)
    
    for i in range(words.shape[0]):
        gradW, gradT, gradWord = crf_gradient(W, T, words[i], labels[i], dimX, dimY)
        avg_grad_W += gradW
        avg_grad_T += gradT
        avg_grad_word[i] = gradWord
    avg_grad_W = avg_grad_W / batch_size
    avg_grad_T = avg_grad_T / batch_size
    avg_grad_word = avg_grad_word / batch_size
        
    grad_W = -C * avg_grad_W.T + W
    grad_T = -C * avg_grad_T + T
    grad_W = grad_W * (grad_output)
    grad_T = grad_T * (grad_output)
    avg_grad_word = (-C * avg_grad_word) * (grad_output)
    
    # calculate the gradient for the loss function
    return grad_W, grad_T, avg_grad_word, None, None, None, None

#%%


class CRF(nn.Module):

    def __init__(self, input_dim=(16,8), conv_layers=[[5, 2, 1]], num_labels=26, C=1000):
        """
        Linear chain CRF with convolution layers
        @param input_dim: The dimension of input image. (default (16,8))
        @param conv_layers: A list of params to Conv layers. Each layer has [kernel_size, padding, stride] parameters
        @param num_labels: Number of labels in raw input (default: 26)
        @param C: The complexity parameter for regularization. (default: 1000)
        """
        super(CRF, self).__init__()
        self.input_dim = input_dim 
        self.num_labels = num_labels
        self.C = C
        # Temp Conv Impl: #TODO Remove this and handle multiple layers
        self.conv_layers = []
        if conv_layers is not None and len(conv_layers) != 0:
            for kernel_size, zero_padding, stride in conv_layers:
                self.conv_layers.append(nn.Conv2d(in_channels=1, out_channels=1,
                       kernel_size=kernel_size, stride=stride, 
                       padding=zero_padding, bias=False)
                    )
        for i,layer in enumerate(self.conv_layers):
            name = "layer{}".format(i+1)
            self.add_module(name, layer)
        
        # Replicating old CRF on pytorch
        self.init_params()
        self.use_cuda = torch.cuda.is_available()

        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]
            
            
    def get_embedding_dim(self):
        """
        Automatically calculate the dimension of the embedding produced by conv layers.
        The flatten embedding will be sent to the CRF layer. So, the parameters should be
        initialized using embedding dimension.
        Returns
        -------
        tuple (emb_dimX, emb_dimY)
        """
        input_dim = torch.tensor(self.input_dim)
        for layer in self.conv_layers:
            input_dim = 1 + (input_dim + 2*torch.tensor(layer.padding) - \
                             (torch.tensor(layer.kernel_size) -1) \
                            - 1)//torch.tensor(layer.stride)
        return input_dim
    
    def init_params(self):
        """
        Initialize trainable parameters of CRF here
        """ 
        embed_dim = self.get_embedding_dim()
        self.embed_dim = torch.prod(embed_dim).item()
        self.W = Parameter(torch.empty(self.embed_dim, self.num_labels), requires_grad=True)
        self.T = Parameter(torch.empty(self.num_labels, self.num_labels), requires_grad=True)
        self.W.data.zero_()
        self.T.data.zero_()
        
          
    def load_params(self, W, T):
        if W.shape[0] == self.embed_dim and W.shape[1] == self.num_labels  \
            and T.shape[0] == self.num_labels and T.shape[1] == self.num_labels:
            self.W.data = W
            self.T.data = T
        else:
            raise Exception("The dimension of parameters do not match. Expected W: {} x {} and T : {} x {}".format(
                            self.embed_dim, self.num_labels, self.num_labels, self.num_labels))

    def forward(self, X):
        """
        Implement the objective of CRF here.
        The input (features) to the CRF module should be convolution features.
        """
        with torch.no_grad():
            features = self.get_conv_features(X) 
            batch_size, max_chars, dimX = features.shape
            dimY = self.num_labels
            dimX = self.embed_dim
            predictions = torch.zeros(batch_size, max_chars, dimY)
            for i,word in enumerate(features):
              preds = crf_decode(self.W,self.T,word,dimX,dimY)
              one_hot_preds = torch.zeros(max_chars, dimY, dtype=torch.long)
              one_hot_preds[torch.arange(len(preds)), preds] = 1
              predictions[i] = one_hot_preds
            return predictions

    def loss(self, words, labels):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        features = self.get_conv_features(words)
        dimX = self.embed_dim
        dimY = self.num_labels
        # Commented code for utilizing pytorch autograd and checking grads
        # W = self.W
        # T = self.T
        # C = self.C

        # log_crf = 0
        # for i in range(features.shape[0]):
        #     log_crf += crf_logloss(W, T, features[i], labels[i], dimX, dimY)
        
        # log_crf = log_crf / features.shape[0]
        # # print(log_crf)
        # f = (-C *log_crf)  + 0.5 * torch.sum(W*W) + 0.5 * torch.sum(T*T)
        # return f

        return CRFLoss.apply(self.W, self.T, features, labels, self.C, dimX, dimY)
    


    def get_conv_features(self, X):
        """
        Generate convolution features for a given word
        """
        # TODO Handle multiple Conv layers
        if self.conv_layers is None or len(self.conv_layers) == 0:
            return X
        else:
            batch_size, max_chars, dimX = X.shape
            all_images = X.view(-1,1,self.input_dim[0],self.input_dim[1])
            for layer in self.conv_layers:
                out = layer(all_images)
            out1 = out.view(batch_size, max_chars, -1)
            assert out1.shape[2] == self.embed_dim
            return out1
#%%

    