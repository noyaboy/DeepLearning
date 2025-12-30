import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 

np.set_printoptions(
    precision=2,
    suppress=True,
    linewidth=300,
    threshold=1000
)

class _Layer(object):
  def __init__(self):
    pass

  def forward(self, *input):
    r"""Define the forward propagation of this layer.

    Should be overridden by all subclasses.
    """
    raise NotImplementedError

  def backward(self, *output_grad):
    r"""Define the backward propagation of this layer.

    Should be overridden by all subclasses.
    """
    raise NotImplementedError

## by yourself .Finish your own NN framework
class Block(_Layer):
  def __init__(self, oc_size, ic_size):
    self.oc_size = oc_size
    self.ic_size = ic_size
    
    self.BatchNorm1 = BatchNorm(c_size = self.ic_size)
    self.SiLU1  = SiLU()
    self.Convolution1 = Convolution(oc_size = self.oc_size, ic_size = self.ic_size, kernel_size = 3, stride = 2, padding = 1)
    self.BatchNorm2 = BatchNorm(c_size = self.oc_size)
    self.SiLU2  = SiLU()
    self.Convolution2 = Convolution(oc_size = self.oc_size, ic_size = self.oc_size, kernel_size = 3, stride = 1, padding = 1)

    self.ConvolutionResidual = Convolution(oc_size = self.oc_size, ic_size = self.ic_size, kernel_size = 1, stride = 2, padding = 0)

    self.BatchNorm3 = BatchNorm(c_size = self.oc_size)
    self.SiLU3  = SiLU()
    self.Convolution3 = Convolution(oc_size = self.oc_size, ic_size = self.oc_size, kernel_size = 3, stride = 1, padding = 1)
    self.BatchNorm4 = BatchNorm(c_size = self.oc_size)
    self.SiLU4  = SiLU()
    self.Convolution4 = Convolution(oc_size = self.oc_size, ic_size = self.oc_size, kernel_size = 3, stride = 1, padding = 1)
    
    # zero initialization
    self.BatchNorm2.gamma[...] = 0.0
    self.BatchNorm4.gamma[...] = 0.0

    self.res_scale = 1.0

  def forward(self, input, training: bool = True):
    input_main = input
    input_residual = input

    output_main = self.BatchNorm1.forward(input_main, training = training)
    output_main = self.SiLU1.forward(output_main)
    output_main = self.Convolution1.forward(output_main)

    output_main = self.BatchNorm2.forward(output_main, training = training)
    output_main = self.SiLU2.forward(output_main)
    output_main = self.Convolution2.forward(output_main)

    output_residual = self.ConvolutionResidual.forward(input_residual)

    input_main = output_residual + self.res_scale * output_main
    input_residual = input_main

    output_main = self.BatchNorm3.forward(input_main, training = training)
    output_main = self.SiLU3.forward(output_main)
    output_main = self.Convolution3.forward(output_main)

    output_main = self.BatchNorm4.forward(output_main, training = training)
    output_main = self.SiLU4.forward(output_main)
    output_main = self.Convolution4.forward(output_main)

    output_residual = input_residual

    output = output_residual + self.res_scale * output_main

    return output

  def backward(self, output_grad):
    output_grad_main = self.res_scale * output_grad
    output_grad_residual = output_grad

    input_grad_main = self.Convolution4.backward(output_grad_main)
    input_grad_main = self.SiLU4.backward(input_grad_main)
    input_grad_main = self.BatchNorm4.backward(input_grad_main)

    input_grad_main = self.Convolution3.backward(input_grad_main)
    input_grad_main = self.SiLU3.backward(input_grad_main)
    input_grad_main = self.BatchNorm3.backward(input_grad_main)

    input_grad_residual = output_grad_residual

    output_grad_total = input_grad_main + input_grad_residual
    
    output_grad_residual = output_grad_total
    output_grad_main = self.res_scale * output_grad_total

    input_grad_main = self.Convolution2.backward(output_grad_main)
    input_grad_main = self.SiLU2.backward(input_grad_main)
    input_grad_main = self.BatchNorm2.backward(input_grad_main)

    input_grad_main = self.Convolution1.backward(input_grad_main)
    input_grad_main = self.SiLU1.backward(input_grad_main)
    input_grad_main = self.BatchNorm1.backward(input_grad_main)

    input_grad_residual = self.ConvolutionResidual.backward(output_grad_residual)

    input_grad = input_grad_main + input_grad_residual

    return input_grad

## by yourself .Finish your own NN framework
class Convolution(_Layer):
  def __init__(self, oc_size, ic_size, kernel_size, stride, padding, dtype = np.float32):
    self.stride = stride
    self.padding = padding
    self.oc_size = oc_size
    self.ic_size = ic_size
    self.kh_size = kernel_size
    self.kw_size = kernel_size
    self.dtype = dtype

    self.fan_in = self.ic_size * self.kh_size * self.kw_size

    self.kernel = np.random.randn(self.oc_size, self.ic_size, self.kh_size, self.kw_size).astype(dtype) * np.sqrt( 2.0 / self.fan_in ).astype(dtype)
    self.bias = np.zeros(self.oc_size, dtype = dtype)

    self.kernel_v = np.zeros_like(self.kernel)
    self.bias_v = np.zeros_like(self.bias)

    self.bias_grad = np.zeros(self.oc_size, dtype = dtype)
    self.kernel_grad = np.zeros((self.oc_size, self.ic_size, self.kh_size, self.kw_size), dtype = dtype)
    
    self._built = False

  def _build_once(self, input):
    self.n_size, self.ih_size, self.iw_size = input.shape[0], input.shape[2], input.shape[3]
    self.oh_size = (self.ih_size + 2 * self.padding - self.kh_size) // self.stride + 1
    self.ow_size = (self.iw_size + 2 * self.padding - self.kw_size) // self.stride + 1

    oh_idx = self.stride * np.arange(self.oh_size)
    iw_idx = self.stride * np.arange(self.ow_size)
    k_idx  = np.arange(self.kh_size)
    l_idx  = np.arange(self.kw_size)

    self.ih_index = (k_idx[:, None] + oh_idx[None, :]).astype(np.int64)
    self.iw_index = (l_idx[:, None] + iw_idx[None, :]).astype(np.int64)
    self.M = self.oh_size * self.ow_size
    self.K = self.ic_size * self.kh_size * self.kw_size

    self._built = True

  def _im2col(self, input_pad):
    # n, ic, ih+kh-1, iw+kw-1, kh, kw
    window = sliding_window_view(input_pad, (self.kh_size, self.kw_size), axis=(-2, -1))
    # n(n), ic(c), oh(h), ow(w), kh(k), kw(l)
    window = window[:, :, ::self.stride, ::self.stride, :, :]

    # -> n(n), oh(h), ow(w), ic(c), kh(k), kw(l)
    # -> n, M (oh * ow), K (ic * kh * kw)
    col = window.transpose(0, 2, 3, 1, 4, 5).reshape(input_pad.shape[0], self.M, self.K)
    return col

  def forward(self, input):
    if input.dtype != self.dtype:
      input = input.astype(self.dtype, copy = False)

    if self._built == False:
      self._build_once(input)

    # # Only need building once
    # self.n_size, self.ih_size, self.iw_size = input.shape[0], input.shape[2], input.shape[3]
    # self.oh_size = (self.ih_size + 2 * self.padding - self.kh_size) // self.stride + 1
    # self.ow_size = (self.iw_size + 2 * self.padding - self.kw_size) // self.stride + 1
 
    self.last_input = input

    if self.padding > 0:
      input_pad = np.pad(input, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant', constant_values=0)
    else:
      input_pad = input

    # Change to Im2col + GEMM
    # # n, ic, ih+kh-1, iw+kw-1, kh, kw
    # window = sliding_window_view(input_pad, (self.kh_size, self.kw_size), axis=(-2, -1))
    # # n(n), ic(c), oh(h), ow(w), kh(k), kw(l)
    # window = window[:, :, ::self.stride, ::self.stride, :, :]
    
    # output = np.einsum('n c h w k l, o c k l -> n o h w',window, self.kernel)
    # output += self.bias[None, :, None, None]

    # n, M(oh * ow), K(ic * kh * kw)
    col = self._im2col(input_pad)
    self._last_col = col
    
    # oc, ic, kh, kl 
    # -> oc, K(ic * kh * kw)
    # -> K(ic * kh * kw), oc
    weight_col = self.kernel.reshape(self.oc_size, self.K).T
    output_col = col @ weight_col

    # n, M(oh * ow), oc
    # -> n, oh, ow, oc
    output = output_col.reshape(input.shape[0], self.oh_size, self.ow_size, self.oc_size)

    # -> n, oc, oh, ow
    output = output.transpose(0, 3, 1, 2).astype(self.dtype, copy = False)
    output += self.bias[None, :, None, None]

    return output

  def backward(self, output_grad):
    output_grad = output_grad.astype(self.dtype, copy = False)
    self.bias_grad[...] = output_grad.sum( axis = (0, 2, 3) )
    
    batch_size = output_grad.shape[0]

    # n, oc, oh, ow
    # -> n, oh, ow, oc
    # -> n, M(oh, ow), oc
    output_grad_col = output_grad.transpose(0, 2, 3, 1).reshape(batch_size, self.oh_size * self.ow_size, self.oc_size)    
    
    # [ n, M(oh, ow), oc ] * [ n, M(oh * ow), K(ic * kh * kw) ]
    self.kernel_grad[...] = np.tensordot(output_grad_col, self._last_col, axes = ([0, 1], [0, 1])) \
        .reshape(self.oc_size, self.ic_size, self.kh_size, self.kw_size) \
        .astype(self.dtype, copy = False)

    # oc, ic, kh, kl 
    # -> oc, K(ic * kh * kw)
    weight_col = self.kernel.reshape(self.oc_size, self.K)

    # -> n, M(oh, ow), K(ic * kh * kw)
    input_grad_col = output_grad_col @ weight_col
    # -> n, oh, ow, ic, kh, kw
    input_grad_col = input_grad_col.reshape(batch_size, self.oh_size, self.ow_size, self.ic_size, self.kh_size, self.kw_size)
    # -> n, ic, kh, kw, oh, ow
    input_grad_col = input_grad_col.transpose(0, 3, 4, 5, 1, 2)

    # Change to Im2col + GEMM
    # input = self.last_input
    # input_pad = np.pad(input, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant', constant_values=0)
    # # window: n, ic, ih+kh-1, iw+kw-1, kh, kw
    # window = sliding_window_view(input_pad, (self.kh_size, self.kw_size), axis=(-2, -1))
    # # n(n), ic(c), oh(h), ow(w), kh(k), kw(l)
    # window = window[:, :, ::self.stride, ::self.stride, :, :]
    # self.kernel_grad = np.einsum('n c h w k l, n o h w -> o c k l', window, output_grad)
    # # (n, ic, oh, ow, kh, kw)
    # window_grad = np.einsum('n o h w, o c k l -> n c h w k l', output_grad, self.kernel)
    # # (n, ic, kh, kw, oh, ow)
    # window_grad_transpose = window_grad.transpose(0, 1, 4, 5, 2, 3)

    if self.padding > 0:
      ih_pad_size = self.ih_size + 2 * self.padding
      iw_pad_size = self.iw_size + 2 * self.padding
    else:
      ih_pad_size, iw_pad_size = self.ih_size, self.iw_size

    input_pad_grad = np.zeros((batch_size, self.ic_size, ih_pad_size, iw_pad_size), dtype = self.dtype)

    # # Too Slow ...
    # for k in range(self.kh_size):
    #   ih_index = k + self.stride * np.arange(self.oh_size)
    #   for l in range(self.kw_size):
    #     iw_index = l + self.stride * np.arange(self.ow_size)
    #     np.add.at(
    #       input_pad_grad, 
    #       (slice(None), slice(None), ih_index[:, None], iw_index[None, :]),
    #       window_grad[..., k, l]  
    #     )

    # # Only need building once
    # oh_idx = self.stride * np.arange(self.oh_size)
    # iw_idx = self.stride * np.arange(self.ow_size)
    # k_idx  = np.arange(self.kh_size)
    # l_idx  = np.arange(self.kw_size)
    # self.ih_index = k_idx[:, None] + oh_idx[None, :]
    # self.iw_index = l_idx[:, None] + iw_idx[None, :]

    # np.add.at(
    #     input_pad_grad,
    #     (slice(None), slice(None), self.ih_index[:, None, :, None], self.iw_index[None, :, None, :]),
    #     # window_grad_transpose
    #     # (n, ic, kh, kw, oh, ow) to (n, ic, ih_pad, iw_pad)
    #     input_grad_col
    # )

    for it_k in range(self.kh_size):
      ih_slice = slice(it_k, it_k + self.oh_size * self.stride, self.stride)
      for it_l in range(self.kw_size):
        iw_slice = slice(it_l, it_l + self.ow_size * self.stride, self.stride)
        # broadcast: *n, *ic, kh, kw, *oh, *ow
        input_pad_grad[:, :, ih_slice, iw_slice] += input_grad_col[:, :, it_k, it_l, :, :]

    if self.padding > 0:
      input_grad = input_pad_grad[:, :, self.padding: -self.padding, self.padding: -self.padding]
    else:
      input_grad = input_pad_grad

    return input_grad

## by yourself .Finish your own NN framework
class FullyConnected(_Layer):
  def __init__(self, input_size, output_size, dtype = np.float32):
    self.dtype = dtype
    self.input_size = input_size
    self.output_size = output_size
    self.weight = np.random.randn(output_size, input_size).astype(dtype) * np.sqrt(2.0 / self.input_size).astype(dtype)
    self.bias = np.zeros(output_size, dtype = dtype)

    self.weight_v = np.zeros_like(self.weight)
    self.bias_v = np.zeros_like(self.bias)

    self.weight_grad = np.zeros((output_size, input_size), dtype = dtype)
    self.bias_grad = np.zeros(output_size, dtype = dtype)

  def forward(self, input, training: bool = True):
    self.training = training
    self.input = input.astype(self.dtype, copy = False)
    output = self.input @ self.weight.T + self.bias[None, :]

    return output

  def backward(self, output_grad):
    output_grad = output_grad.astype(self.dtype, copy = False)

    self.bias_grad[...] = output_grad.sum(axis = 0) 
    self.weight_grad[...] = output_grad.T @ self.input  
    input_grad = output_grad @ self.weight

    return input_grad

## by yourself .Finish your own NN framework
class SiLU(_Layer):
  def __init__(self):
    pass

  def forward(self, input):
    self.input = input
    self.sigmoid = 0.5 * (1.0 + np.tanh(0.5 * input))
    output = input * self.sigmoid
    return output

  def backward(self, output_grad):
    input_grad = output_grad * (self.sigmoid + self.input * self.sigmoid * (1.0 - self.sigmoid))
    return input_grad

## by yourself .Finish your own NN framework
class GlobalAveragePooling(_Layer):
  def __init__(self):
    pass

  def forward(self, input):
    self.n_size, self.c_size, self.h_size, self.w_size = input.shape
    self.number_hw = self.h_size * self.w_size
    
    output = np.einsum('nchw -> nc', input) / self.number_hw
    
    return output

  def backward(self, output_grad):
    ones_hw = np.ones((self.h_size, self.w_size)) 
    input_grad = np.einsum('nc, hw-> nchw', output_grad, ones_hw) / self.number_hw 
    return input_grad

class SoftmaxWithloss(_Layer):
  def __init__(self, epsilon: float = 1e-12, smoothing: np.float32 = 0.05):
    self.epsilon = epsilon
    self.smoothing = smoothing

  def forward(self, input, target, training: bool = True):
    self.target = target
    self.training = training
    self.n_size = input.shape[0]
    input_shift = input - input.max(axis = 1, keepdims=True)
    input_shift_exp = np.exp(input_shift)    
    denom = np.einsum('n i -> n', input_shift_exp)
    
    if self.training == True:
      if self.smoothing > 0.0:
        target = target * (1 - self.smoothing) + self.smoothing / 10

    self.predict = input_shift_exp / ( denom[:, None] + self.epsilon )

    self.predict_log = np.log(self.predict + self.epsilon)
    loss_per_sample = -np.einsum('n i, n i -> n', self.predict_log, self.target)

    loss = np.einsum('n ->', loss_per_sample) / self.n_size

    return self.predict, loss

  def backward(self):
    input_grad = (self.predict - self.target) / self.n_size
    
    return input_grad
    

## by yourself .Finish your own NN framework
class BatchNorm(_Layer):
  def __init__(self, c_size, epsilon: float = 1e-5, momentum: float = 0.9, dtype = np.float32):
    self.c_size = c_size
    self.epsilon = float(epsilon)
    self.momentum = float(momentum)
    self.dtype = dtype

    self.gamma = np.ones(self.c_size, dtype = dtype)
    self.beta = np.zeros(self.c_size, dtype = dtype)

    self.gamma_v = np.zeros_like(self.gamma)
    self.beta_v = np.zeros_like(self.beta)

    self.gamma_grad = np.zeros(self.c_size, dtype = dtype)
    self.beta_grad = np.zeros(self.c_size, dtype = dtype)
    
    self.moving_mean = np.zeros(self.c_size, dtype = dtype)
    self.moving_var = np.ones(self.c_size, dtype = dtype)

  def _bcastChannel(self, input):
    return input[None, :, None, None]

  def forward(self, input, training: bool = True):
    self.n_size, _, self.ih_size, self.iw_size = input.shape
    self.number = float(self.n_size * self.ih_size * self.iw_size)
    self.training = training
    input = input.astype(self.dtype, copy = False)

    if self.training == True:
      mean = np.einsum('nchw -> c', input) / self.number
      self.input_centered = input - self._bcastChannel(mean) 
      var = np.einsum('nchw, nchw-> c', self.input_centered, self.input_centered) / self.number
      var = np.maximum(var, 0.0)
      self.inv_std = 1.0 / np.sqrt(var + self.epsilon)
      self.input_hat = self.input_centered * self._bcastChannel(self.inv_std)
      output = self._bcastChannel(self.gamma) * self.input_hat + self._bcastChannel(self.beta)

      self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
      self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * var
    else:
      mv = np.maximum(self.moving_var, 0.0)
      inv_std = 1.0 / np.sqrt(mv + self.epsilon)
      input_centered = input - self._bcastChannel(self.moving_mean)
      input_hat = input_centered * self._bcastChannel(inv_std)

      output = self._bcastChannel(self.gamma) * input_hat + self._bcastChannel(self.beta)
    
    return output

  def backward(self, output_grad):
    output_grad = output_grad.astype(self.dtype, copy = False)
    self.beta_grad  = np.einsum('nchw->c', output_grad)
    self.gamma_grad = np.einsum('nchw,nchw->c', output_grad, self.input_hat)

    gamma_bc = self._bcastChannel(self.gamma)
    invstd_bc = self._bcastChannel(self.inv_std)

    dx_hat = output_grad * gamma_bc

    sum_dxhat = self._bcastChannel(np.einsum('nchw->c', dx_hat)) / self.number
    sum_dxhat_xhat = self._bcastChannel(np.einsum('nchw,nchw->c', dx_hat, self.input_hat)) / self.number

    input_grad = invstd_bc * (dx_hat - sum_dxhat - self.input_hat * sum_dxhat_xhat)
    return input_grad.astype(self.dtype, copy = False)




















  