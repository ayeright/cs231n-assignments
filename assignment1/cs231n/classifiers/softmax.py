import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
      # compute class scores
      scores = X[i].dot(W)
      # shift them so max is 0 to improve numerical stability
      # this won't change loss
      scores += -np.max(scores)
      # sum the exponentials for each class
      exp_sum = np.sum(np.exp(scores))
      # compute the probability for each class
      p = lambda j: np.exp(scores[j]) / exp_sum
      # compute loss
      loss += -np.log(p(y[i]))
      #compute gradients
      for j in range(num_classes):
          p_j = p([j])
          dW[:, j] += (p_j - (j == y[i])) * X[i]
       
  # average over training set    
  loss /= num_train
          
  # add regularisation loss
  loss += 0.5 * reg * np.sum(W * W)
  
  # average gradient over training set
  dW /= num_train
          
  # add regularisation gradient
  dW += reg * W
          
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
 
  #compute loss  
  
  # compute class scores
  f = X.dot(W)
  # shift them so max is 0 to improve numerical stability
  # this won't change loss
  f += -np.max(f, axis=1)[:, None]
  # compute loss
  f_y = f[np.arange(f.shape[0]), y]
  exp_sum = np.sum(np.exp(f), axis=1)[:, None]
  p_y = np.exp(f_y) / exp_sum
  loss = np.mean(-np.log(p_y))
  # add regularisation loss
  loss += 0.5 * reg * np.sum(W * W)
  
  #compute gradient
  
  # compute the probability for every class in every row
  p = np.exp(f) / exp_sum
  # create a matrix of zeros 
  # with a one on each row at the position of correct class
  M = np.zeros((num_train, num_classes))
  M[np.arange(num_train), y] = 1
  # dot product with X
  dW += X.T.dot(p - M) 
  dW /= num_train 
  # add regularisation gradient
  dW += reg * W
   
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

