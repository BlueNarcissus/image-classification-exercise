import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg=0.0):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)   # (D,K)
    
    num_train=X.shape[0]
    num_classes=W.shape[1]
    
    scores = np.dot(X,W)    # (N.K)
    scores -= np.max(scores, axis=1).reshape(-1,1)
    exp_scores = np.exp(scores)
    dscores = np.zeros_like(scores)
    for i in range(num_train):
        probs = exp_scores[i] / np.sum(exp_scores[i])   # (1,K)
        correct_logprobs = -np.log(probs[y[i]])
        loss += np.sum(correct_logprobs)
        dscores[i] = probs
        dscores[i, y[i]] -= 1
            
    loss = loss/num_train + 0.5*reg*np.sum(W*W)
    dW = np.dot(X.T, dscores)/num_train
    dW += reg*W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg=0.0):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    num_train=X.shape[0]
    num_classes=W.shape[1]
    
    scores = np.dot(X,W)
    scores -= np.max(scores, axis=1).reshape(-1,1)
    probs = np.exp(scores)
    probs = probs/np.sum(probs,axis=1).reshape(-1,1)
    correct_logprobs = -np.log(probs[xrange(num_train),y])
    loss = np.sum(correct_logprobs)/num_train + 0.5*reg*np.sum(W*W)
    
    dscores = probs
    dscores[xrange(num_train),y] -= 1
    dscores /= num_train
    dW = np.dot(X.T, dscores)
    dW += reg*W

    return loss, dW


