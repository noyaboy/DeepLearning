import numpy as np
from .layer import *

class Network(object):
  def __init__(self, config = [64, 128, 256]):
    
    self.ConvolutionStem = Convolution(oc_size = config[0], ic_size = 1, kernel_size = 3, stride = 1, padding = 1)
    self.Block1 = Block(oc_size = config[0], ic_size = config[0])
    self.Block2 = Block(oc_size = config[1], ic_size = config[0])
    self.Block3 = Block(oc_size = config[2], ic_size = config[1])
    self.GlobalAveragePooling = GlobalAveragePooling()
    self.FullyConnected = FullyConnected(input_size = config[2], output_size = 10)
    self.SoftmaxWithloss = SoftmaxWithloss()



  def _init_velocity(self, param):
    if not hasattr(param, "v"):
      param.v = np.zeros_like(param)

  def forward(self, input, target, training: bool = True):
    ## by yourself .Finish your own NN framework
    input_reshaped = input.reshape(input.shape[0], 1, int(np.sqrt(input.shape[1])), int(np.sqrt(input.shape[1])))

    if training:
      for i in range(input_reshaped.shape[0]):
        if np.random.rand() < 0.5:
          input_reshaped[i] = np.flip(input_reshaped[i], axis = -1)

        if np.random.rand() < 0.5:
          input_reshaped_padded = np.pad(input_reshaped[i], ((0,0), (2, 2), (2, 2)), mode='constant', constant_values=0)

          windows = np.lib.stride_tricks.sliding_window_view(
            input_reshaped_padded, window_shape=(28, 28), axis=(1, 2)
          )

          random_ih = int(np.random.rand() * 5)
          random_iw = int(np.random.rand() * 5)

          input_reshaped[i] = windows[:, random_ih, random_iw]

    output = self.ConvolutionStem.forward(input_reshaped)
    output = self.Block1.forward(output, training = training)
    output = self.Block2.forward(output, training = training)
    output = self.Block3.forward(output, training = training)
    output = self.GlobalAveragePooling.forward(output)
    output = self.FullyConnected.forward(output, training = training)
    pred, loss = self.SoftmaxWithloss.forward(output, target, training = training)

    return pred, loss

  def backward(self):
    ## by yourself .Finish your own NN framework

    input_grad = self.SoftmaxWithloss.backward()
    input_grad = self.FullyConnected.backward(input_grad)
    input_grad = self.GlobalAveragePooling.backward(input_grad)
    input_grad = self.Block3.backward(input_grad)
    input_grad = self.Block2.backward(input_grad)
    input_grad = self.Block1.backward(input_grad)
    _ = self.ConvolutionStem.backward(input_grad)

  def update(self, lr, wd = 3e-4, momentum = 0.9):
    ## by yourself .Finish your own NN framework
    for conv in [
        self.ConvolutionStem,
        self.Block1.Convolution1, self.Block1.Convolution2, self.Block1.ConvolutionResidual,
        self.Block1.Convolution3, self.Block1.Convolution4,
        self.Block2.Convolution1, self.Block2.Convolution2, self.Block2.ConvolutionResidual,
        self.Block2.Convolution3, self.Block2.Convolution4,
        self.Block3.Convolution1, self.Block3.Convolution2, self.Block3.ConvolutionResidual,
        self.Block3.Convolution3, self.Block3.Convolution4,
    ]:

      conv.kernel_grad += wd * conv.kernel
      conv.kernel_v = momentum * conv.kernel_v - lr * conv.kernel_grad
      conv.kernel += conv.kernel_v

      conv.bias_v = momentum * conv.bias_v - lr * conv.bias_grad
      conv.bias += conv.bias_v

    # batch norms
    for bn in [
        self.Block1.BatchNorm1, self.Block1.BatchNorm2, self.Block1.BatchNorm3, self.Block1.BatchNorm4,
        self.Block2.BatchNorm1, self.Block2.BatchNorm2, self.Block2.BatchNorm3, self.Block2.BatchNorm4,
        self.Block3.BatchNorm1, self.Block3.BatchNorm2, self.Block3.BatchNorm3, self.Block3.BatchNorm4,
    ]:
      bn.gamma_v = momentum * bn.gamma_v - lr * bn.gamma_grad
      bn.gamma += bn.gamma_v

      bn.beta_v = momentum * bn.beta_v - lr * bn.beta_grad
      bn.beta += bn.beta_v

    # fully connected
    self.FullyConnected.weight_grad += wd * self.FullyConnected.weight
    self.FullyConnected.weight_v = momentum * self.FullyConnected.weight_v - lr * self.FullyConnected.weight_grad
    self.FullyConnected.weight += self.FullyConnected.weight_v

    self.FullyConnected.bias_v = momentum * self.FullyConnected.bias_v - lr * self.FullyConnected.bias_grad
    self.FullyConnected.bias += self.FullyConnected.bias_v
        
