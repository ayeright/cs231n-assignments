import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  num_train = x.shape[0]
  dim = np.prod(x.shape[1:])
  x_2d = x.reshape(num_train, dim) # shape (N x D)
  out = np.dot(x_2d, w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  
  # backprop out = X.dot(W) + b
  
  # through X
  dx_2d = np.dot(dout, w.T) # shape (N, D)
  dx = dx_2d.reshape(x.shape) # shape (N, d_1, ..., d_k)
  
  # through w
  x_2d = x.reshape(x.shape[0], np.prod(x.shape[1:])) # shape (N, D)
  dw = np.dot(x_2d.T, dout) # shape (D, M)
  
  # through b
  db = np.sum(dout, axis=0) # shape (M,)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  
  # gradient of ReLU wrt x is 1(x > 0)
  dx = (x > 0) * dout  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################

    cache = (x,)
    
    '''
    # zero-centre data
    sums = np.sum(x, axis=0, keepdims=True) # step 1
    mu = sums / N # step 2
    x_zero_mean = x - mu # step 3
    
    cache += (sums, mu, x_zero_mean)
    
    # normalise data
    x_zero_mean_sq = x_zero_mean**2 # step 4
    tot_ss = np.sum(x_zero_mean_sq, axis=0, keepdims=True) # step 5
    var = tot_ss / N + eps # step 6
    std = np.sqrt(var) # step 7
    inv_std = 1. / std # step 8
    x_norm = x_zero_mean * inv_std # step 9
    
    cache += (x_zero_mean_sq, tot_ss, var, std, inv_std, x_norm)  
              
    # scale and shift
    x_norm_scaled = x_norm * gamma # step 10
    out = x_norm_scaled + beta # step 11
    
    cache += (x_norm_scaled, gamma, beta)
    
    '''   
    
    # zero-centre data
    x_mean = np.sum(x, axis=0, keepdims=True) / N # step 1
    cache += (x_mean,)  
    x_zero_mean = x - x_mean # step 2
    cache += (x_zero_mean,)
    
    # normalise data
    x_var = np.sum(x_zero_mean**2, axis=0, keepdims=True) / N + eps # step 3
    cache += (eps, x_var)
    x_std = np.sqrt(x_var) # step 4
    cache += (x_std,)
    x_norm = x_zero_mean / x_std # step 5
    cache += (x_norm,)
    
    # scale and shift
    out = x_norm * gamma + beta # step 6
    cache += (gamma, beta)
    
    mu = x_mean
    var = x_var
    
    
    # update running mean and variance
    running_mean = momentum * running_mean + (1 - momentum) * mu
    running_var = momentum * running_var + (1 - momentum) * var
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    running_std = np.sqrt(running_var)
    x_norm = (x - running_mean) / (running_std + eps)
    out = x_norm * gamma + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  N, D = dout.shape 
  
  '''
  # unpack cache
  (x, sums, mu, x_zero_mean, x_zero_mean_sq, tot_ss, var, std,
   inv_std, x_norm, x_norm_scaled, gamma, beta) = cache
   
  # backprop step 11: out = x_norm_scaled + beta
  dx_norm_scaled = (1.) * dout
  dbeta = np.sum((1.) * dout, axis=0, keepdims=True)
   
  # backprop step 10: x_norm_scaled = x_norm * gamma
  dx_norm = (gamma) * dx_norm_scaled
  dgamma = np.sum((x_norm) * dx_norm_scaled, axis=0, keepdims=True)
   
  # backprop step 9: x_norm = x_zero_mean * inv_std
  dx_zero_mean = (inv_std) * dx_norm
  dinv_std = np.sum((x_zero_mean) * dx_norm, axis=0, keepdims=True)
   
  # backprop step 8: inv_std = 1. / std
  dstd = (-1. / (std**2)) * dinv_std
     
  # backprop step 7: std = np.sqrt(var)
  dvar = (0.5 / (var**0.5)) * dstd  
     
  # backprop step 6: var = tot_ss / N + eps
  dtot_ss = (1. / N) * dvar   
     
  # backprop step 5: tot_ss = np.sum(x_zero_mean_sq, axis=0, keepdims=True)
  dx_zero_mean_sq = (np.ones_like(x_zero_mean_sq)) * dtot_ss  
     
  # backprop step 4: x_zero_mean_sq = x_zero_mean**2
  dx_zero_mean += (2. * x_zero_mean) * dx_zero_mean_sq   
     
  # backprop step 3: x_zero_mean = x - mu
  dx = (1.) * dx_zero_mean
  dmu = np.sum((-1.) * dx_zero_mean, axis=0, keepdims=True)
     
  # backprop step 2: mu = sums / N
  dsums = (1. / N) * dmu   
     
  # backprop step 1: sums = np.sum(x, axis=0, keepdims=True)
  dx += (np.ones_like(x)) * dsums 
     
  ''' 
  
  # unpack cache 
  x, x_mean, x_zero_mean, eps, x_var, x_std, x_norm, gamma, beta = cache
  
  # backprop step 6: out = x_norm * gamma + beta
  dx_norm = gamma * dout # shape (N, D)
  dgamma = np.sum(x_norm * dout, axis=0) # shape (D,)
  dbeta = np.sum(dout, axis=0) # shape (D,)
  
  # backprop step 5: x_norm = x_zero_mean / x_std
  dx_zero_mean = (1. / x_std ) * dx_norm # shape (N, D)
  dx_std = np.sum((-x_zero_mean / (x_std**2)) * dx_norm, axis=0, keepdims=True) # shape (D,)
  
  # backprop step 4: x_std = np.sqrt(x_var) 
  dx_var = (0.5 / (x_var**0.5)) * dx_std # shape (D,)
  
  # backprop step 3: x_var = np.sum(x_zero_mean**2) / N + eps
  dx_zero_mean += (2. * x_zero_mean / N) * dx_var # shape (N, D)
  
  # backprop step 2: x_zero_mean = x - x_mean
  dx = (1.) * dx_zero_mean # shape (N, D)
  dx_mean = np.sum((-1.) * dx_zero_mean, axis=0, keepdims=True) # shape (D,)
  
  # backprop step 1: x_mean = np.sum(x) / N
  dx += (1. / N) * dx_mean
  
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  N, D = dout.shape  
  
  # unpack cache 
  x, x_mean, x_zero_mean, eps, x_var, x_std, x_norm, gamma, beta = cache  
  
  # backprop out = (x_zero_mean / x_std) * gamma + beta
  dbeta = np.sum(dout, axis=0) # shape (D,)
  dgamma = np.sum((x_zero_mean / x_std) * dout, axis=0, keepdims=True)  
  dx_zero_mean = ((1. / x_std) - (x_zero_mean**2 / (N * (x_std**3)))) * gamma * dout
  
  # backprop x_zero_mean = x - x_mean
  dx = (1.) * dx_zero_mean
  dx_mean = np.sum((-1.) * dx_zero_mean, axis=0, keepdims=True)
  
  # backprop x_mean = np.sum(x) / N
  dx += (1. / N) * dx_mean
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) > p) / (1 - p)
    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x #* (1 - p)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    # backprop out = x * mask
    dx = mask * dout
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape  
  
  stride = conv_param['stride']
  pad = conv_param['pad']
  
  # zero-pad the input in width and height
  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant', constant_values=0)
  
  # calculate the output width and height
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  
  out = np.zeros((N, F, H_out, W_out))
  
  for f in range(F):
      for iy in range(H_out):
          for ix in range(W_out):
              # get the regions interest
              y1 = iy * stride
              y2 = HH + (iy * stride)
              x1 = ix * stride
              x2 = WW + (ix * stride)
                            
              # perform the dot product of the weights with every image at correct region
              out[:, f, iy, ix] = \
              np.sum(x_pad[:, :, y1:y2, x1:x2] * w[f, :, :, :], axis=(1, 2, 3)) + b[f]              
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################

  x, w, b, conv_param = cache
  
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  # x.shape = (N, C, H, W)
  # w.shape = (F, C, HH, WW)
  # dout.shape = (N, F, H_out, W_out)  
  
  stride = conv_param['stride']
  pad = conv_param['pad']
  
  # zero-pad the input in width and height
  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant', constant_values=0)
  
  # calculate the output width and height
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  
  dx_pad = np.zeros_like(x_pad)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  
  for f in range(F):
      for iy in range(H_out):
          for ix in range(W_out):
              # get the regions of interest
              y1 = iy * stride
              y2 = HH + (iy * stride)
              x1 = ix * stride
              x2 = WW + (ix * stride)
                                                        
              # backprop:
              # out[:, f, iy, ix] = np.sum(x_pad[:, :, y1:y2, x1:x2] * w[f, :, :, :], axis=(1, 2, 3)) + b[f] 
              
              # first do dx_pad
              # x_pad[:, :, y1:y2, x1:x2].shape = (N, C, HH WW)

              # w[f, :, :, :].shape = (1, C, HH, WW)              
              # dout[:, f, y, x].shape = (N, 1, 1, 1)
              
              w_here = w[f, :, :, :][None, :, :, :]
              dout_here = dout[:, f, iy, ix][:, None, None, None]
              dx_pad[:, :, y1:y2, x1:x2] += w_here * dout_here
              
              # now do dw
              # w[f, :, :, :].shape = (1, C, HH, WW)
              
              # x_pad[:, :, y1:y2, x1:x2].shape = (N, C, HH, WW)
              # dout[:, f, y, x].shape = (N, 1, 1, 1)
              
              x_pad_here = x_pad[:, :, y1:y2, x1:x2]
              dw[f, :, :, :] += np.sum(x_pad_here * dout_here, axis=0) 
              
              # now do db
              # b[f].shape = (1,)
              
              # dout[:, f, y, x].shape = (N, 1, 1, 1)
              db[f] += np.sum(dout_here)
                                   
  dx = dx_pad[:, :, pad:pad+H, pad:pad+W]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################

  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  
  # calculate the output width and height
  H_out = 1 + (H - pool_height) / stride
  W_out = 1 + (W - pool_width) / stride
  out = np.zeros((N, C, H_out, W_out))
  
  for iy in range(H_out):
      for ix in range(W_out):
          # get the region of interest
          y1 = iy * stride
          y2 = y1 + pool_height
          x1 = ix * stride
          x2 = x1 + pool_width
          
          # max pool at each depth slice
          out[:, :, iy, ix] = np.max(x[:, :, y1:y2, x1:x2], axis=(2, 3))
                        
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """

  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  N, C, H, W = x.shape
  
  # calculate the output width and height
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  H_out = 1 + (H - pool_height) / stride
  W_out = 1 + (W - pool_width) / stride
  
  # x.shape = (N, C, H, W)
  # dout.shape = (N, C, H_out, W_out)
  dx = np.zeros_like(x)
  
  for iy in range(H_out):
      for ix in range(W_out):
          # get the region of interest
          y1 = iy * stride
          y2 = y1 + pool_height
          x1 = ix * stride
          x2 = x1 + pool_width
          
          # backprop:
          # out[:, :, iy, ix] = np.max(x[:, :, y1:y2, x1:x2], axis=(2, 3))
          
          # x[:, :, y1:y2, x1:x2].shape = (N, C, pool_height, pool_width)          
          # out[:, :, iy, ix].shape = (N, C, 1, 1)
          
          x_here = x[:, :, y1:y2, x1:x2]
          dout_here = dout[:, :, iy, ix][:, :, None, None]
          # gradient is 1 for max element in pool, else zero
          mask = np.zeros_like(x_here)
          depth_slice_max = np.amax(x_here, axis=(2,3))[:, :, None, None]
          mask[x_here >= depth_slice_max] = 1
          dx[:, :, y1:y2, x1:x2] += mask * dout_here
                   
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  
  # vanilla batch norm accepts a matrix of shape (N, D)
  # and computes a mean and variance for each of the D dimensions
  # spatial batch norm computes a mean and variance for each of the C channels
  # reshape x to (N * H * W, C)
  N, C, H, W = x.shape
  x_2d = x.transpose((0, 2, 3, 1)).reshape((N * H * W, C))
  out_2d, cache = batchnorm_forward(x_2d, gamma, beta, bn_param)  
  
  # go back to original shape
  out = out_2d.reshape((N, H, W, C)).transpose((0, 3, 1, 2))
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  
  # as in forward pass, just reshape and use vanilla batch norm
  N, C, H, W = dout.shape
  dout_2d = dout.transpose((0, 2, 3, 1)).reshape((N * H * W, C)) 
  dx_2d, dgamma, dbeta = batchnorm_backward(dout_2d, cache)
  dx = dx_2d.reshape((N, H, W, C)).transpose((0, 3, 1, 2))
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
