# -*- coding: utf-8 -*-
import torch
import numpy as np

#%%
def crf_decode(W, T, word, dimX, dimY):
    """
    Dynamic Programming implementation of CRF Decoding with pytorch.
    """
    # TODO handle zeropadding properly
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
            # print(score) # Score to be reported
            ans = torch.argmax(first_term + lookup[i])
            previousAns[0] = ans
        else:
            second_term = T[:, previousAns[-1]]
            ans = torch.argmax(first_term+second_term+lookup[i])
            previousAns.append(ans)
            
    previousAns.reverse()
    return previousAns

#%%
def logsumexp_trick(sum_term, along_axis=True):
    """
    Perform logsumexp trick to handle numeric overflow/underflow
    """
    if len(sum_term.shape) == 1 or not along_axis:
        max_term = torch.max(sum_term)
        logsumexp = max_term + torch.log(torch.sum(torch.exp(sum_term-max_term)))
    else:
        max_term = torch.max(sum_term, axis=1).values
        logsumexp = max_term + torch.log(torch.sum(torch.exp((sum_term.T-max_term).T), axis=1))
    return logsumexp

def crf_logloss(W, T, word, labels, dimX, dimY):
    """
    CRF loss for single word and label using pytorch
    """
    # TODO Handle zero-padding
    char_count = torch.nonzero(word.sum(axis=1)).size(0)
    word = word[:char_count]
    if len(labels.shape) > 1:
        Y = torch.argmax(labels, axis=1)[:char_count]
    else:
        Y = labels
    # calculating forward messages
    alpha = torch.zeros((char_count, dimY))
    first_term = torch.matmul(word, W)
    second_term = T.T
    for i in range(1, char_count):
        sum_term = (first_term[i-1] + alpha[i-1]) + second_term
        alpha[i] = logsumexp_trick(sum_term) 
    # getting logZ from messages
    logZ = logsumexp_trick(first_term[char_count-1]+alpha[char_count-1])
    w_term = torch.sum(W[:,Y].T * word) # $\sum_{j=1}^m {W_{yj} . x_j}$
    t_term = torch.sum(T[Y[:-1], Y[1:]]) #$T_{yj, yj+1}
    value = -logZ + w_term + t_term
    return value
#%%
def crf_marginals(W, T, word, dimX, dimY):
    """
    Calculate the marginals P(y_s|X) and P(y_s, y_s+1|X)
    using forward and backward messages
    returns (mx26, m-1x26x26) marginal distributions for each letter in the word
    """
    # forward and backward message at once
    char_count = torch.nonzero(word.sum(axis=1)).size(0)
    alpha = torch.zeros((char_count, dimY)) # alphas
    lbeta = torch.zeros((char_count, dimY)) # log version of betas
    word = word[:char_count]
    first_term = torch.matmul(word, W)
    second_term_a = T.T
    second_term_b = T
    for i in range(1, char_count):
        sum_term_a = (first_term[i-1] + alpha[i-1]) + second_term_a
        sum_term_b = (first_term[char_count-i] +lbeta[char_count-i]) + second_term_b
        alpha[i] = logsumexp_trick(sum_term_a) 
        lbeta[char_count-i-1] = logsumexp_trick(sum_term_b)

    marginal_Y = torch.zeros((char_count, dimY))
    marginal_Y_Y1 = torch.zeros((char_count-1, dimY, dimY))  
    
    for i in range(char_count):
        sum_term = first_term[i] + alpha[i] + lbeta[i]
        log_marginal_y = sum_term - logsumexp_trick(sum_term)
        marginal_Y[i] = np.exp(log_marginal_y)
        # calculate other marginal dist as well
        if i < char_count-1:
            transition = T # T_{yi, yi+1}
            outer_sum_w = (first_term[i].reshape(-1,1) + first_term[i+1]).reshape(dimY, dimY)
            outer_sum_m = (alpha[i].reshape(-1,1) + lbeta[i+1])
            sum_term_all = outer_sum_w + transition + outer_sum_m
            log_marginal_y_y1 = sum_term_all - logsumexp_trick(sum_term_all, along_axis=False)
            marginal_Y_Y1[i] = np.exp(log_marginal_y_y1)
            # Got Denominator same as Zx , which is correct
    return marginal_Y, marginal_Y_Y1

def get_ind(labels, k):
    """
    Indicator func [label == k]
    """
    return (labels == k).double()
    
            
def crf_gradient(W, T, word, labels, dimX, dimY):
    """
    calculate the gradient for given word with m chars and corresponding true labels
    W and T are initial weights
    TODO matrix implementation
    """
    grad_word_all = torch.zeros(word.shape)
    char_count = torch.nonzero(word.sum(axis=1)).size(0)
    word = word[:char_count]
    if len(labels.shape) > 1:
        Y = torch.argmax(labels, axis=1)[:char_count]
    else:
        Y = labels
    Y = Y[:char_count]
    
    # get the marginals
    marY, marY1 = crf_marginals(W, T, word, dimX, dimY)
    # calculate w_k for all 26 Ws. To do matrix approach
    grad_W = torch.zeros(dimY, dimX)
    for k in range(0, dimY):
        ind_k = get_ind(Y, k)
        marginal_k = marY[:,k]
        grad_k = torch.matmul((ind_k - marginal_k), word)
        grad_W[k] = grad_k
        
    pairs = zip(Y[:-1], Y[1:])
    grad_T = torch.zeros((dimY, dimY))
    for i,pair in enumerate(pairs):
        ind_ij = torch.zeros((dimY,dimY))
        ind_ij[pair] = 1.0
        grad_T += ind_ij - marY1[i]
        
    grad_word = W.T[Y] - marY.matmul(W.T)
    
    grad_word_all[:char_count] = grad_word
    return grad_W, grad_T, grad_word_all

#%%
def compare(test_X, test_Y, preds):
    wordMatch = 0.0
    letterMatch = 0.0
    letterCount = 0.0
    for i,pred in enumerate(preds):
        char_count = torch.nonzero(test_X[i].sum(axis=1)).size(0)
        pred = torch.argmax(pred, axis=1)[:char_count]
        truth = torch.argmax(test_Y[i], axis=1)[:char_count]
        matchingLetters = (pred == truth).sum()
        letterMatch += matchingLetters
        letterCount += char_count
        if matchingLetters == char_count:
            wordMatch += 1
    letterAcc = letterMatch / letterCount
    wordAcc = wordMatch / preds.shape[0]
    print("Letter Accuracy: ", letterAcc)
    print("Word Accuracy: ", wordAcc)
    return letterAcc, wordAcc