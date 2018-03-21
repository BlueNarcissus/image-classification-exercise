import numpy as np
from random import shuffle

""" compute the loss and gradient """

def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape) # initialize the gradient as zero
    loss = 0.0
    num_classes = W.shape[1]
    num_train = X.shape[0]
    delta=1
    
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        count=0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + delta # note delta = 1
            if margin > 0:
                loss += margin
                count=count+1
                dW[:,j]+=X[i].T
        dW[:,y[i]]+=-count*X[i].T

    loss /= num_train   # average loss
    loss += reg * np.sum(W * W) # Add regularization
    dW=dW/num_train
    dW+=reg*W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    delta=1
    
    scores=np.dot(X,W)
    correct_class_scores=scores[xrange(num_train), y]
    margins=scores-correct_class_scores.reshape(-1,1)+delta
    margins[xrange(num_train), y]=0 # correct class loss shoule be 0
    margins=margins*(margins>0)  # filter the positive loss, (N,K)
    
    loss=np.mean(np.sum(margins,axis=1))
    loss+= reg * np.sum(W * W)
    
    dW+=np.dot(X.T, margins>0)
    # a fully vectorized way?
    for i in range(num_train):
        dW[:,y[i]]-=np.sum(margins[i]>0)*X[i].T
    dW=dW/num_train
    dW+=reg*W

    return loss, dW
