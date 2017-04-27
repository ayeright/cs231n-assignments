import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  dRdW = np.zeros(W.shape) # initialise the regularisation gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    dW_xi = np.zeros(W.shape)
    gt_margin = 0
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if i == 0:
          # regularisation gradient
          dRdW[:, j] = reg * W[:, j]
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        gt_margin += 1
        # gradient of loss with respect to w_j
        dW_xi[:, j] = X[i]
    # gradient of loss with respect to w_yi    
    dW_xi[:, y[i]] = -gt_margin * X[i]
    # add gradient for xi to gradient matrix
    dW += dW_xi
    
  # average gradient over training set
  dW /= num_train
  
  # add regularisation gradient
  dW += dRdW

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  # compute the scores for each class in each example
  scores = X.dot(W)
  
  # get the correct class score for each example
  correct_class_scores = scores[np.arange(scores.shape[0]), y]
  correct_class_scores = \
  np.reshape(correct_class_scores, (correct_class_scores.shape[0], 1))
  
  # compute the margins for each class in each example
  margins = scores - correct_class_scores + 1
  margins[margins < 0] = 0
  
  # margins from correct classes don't contribute
  margins[np.arange(margins.shape[0]), y] = 0
   
  # average over all examples
  loss = np.sum(margins) / X.shape[0]
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # compute the gradient with respect to the correct class weights for each example
  gt_margin = margins > 0
  num_gt_margin = np.sum(gt_margin, axis=1) 
  num_gt_margin = \
  np.reshape(num_gt_margin, (num_gt_margin.shape[0], 1))
  correct_class_grads = -X * num_gt_margin
  
  # add the gradients to the correct columns of the gradient matrix
  # gradient vector for example i is added to column y[i]
  
  # define a matrix of zeros of size (num_train x num_classes),
  # and in row i put a 1 in column y[i]
  M = np.zeros((X.shape[0], W.shape[1]))
  M[np.arange(M.shape[0]), y] = 1
  
  # use matrix multiplication to add gradients to correct columns
  dW += correct_class_grads.T.dot(M)
  
  # gradient of loss on x_i with respect to w_j is x_i if class score for j
  # violates margin, otherwise 0
  # use matrix multiplication to add x_i to column j of dW where margin violated
  dW += X.T.dot(gt_margin)
  
  # divide by the number of examples
  dW /= X.shape[0]
  
  # add regularisation gradient
  dW += reg * W
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
