import numpy as np 
import tensorflow as tf
from abc import ABCMeta, abstractmethod

class BaseModel(metaclass=ABCMeta):
  def __init__(self, input_shape,
                     output_shape,
                     loss_func,
                     start_lr,
                     lr_decay_step,
                     lr_decay_rate):

    self.inputs    = tf.placeholder(tf.float32, 
                                    shape = [None] + list(input_shape), 
                                    name='inputs')
    self.refers    = tf.placeholder(tf.float32, 
                                    shape = [None] + list(output_shape), 
                                    name='reference')

    self.lr        = tf.train.exponential_decay(start_lr,
                                                tf.train.get_or_create_global_step(),
                                                lr_decay_step,
                                                lr_decay_rate, 
                                                staircase=True)

    self.global_step = tf.train.get_or_create_global_step()

    self.outputs  = self.build_model()
    self.loss     = self.build_loss(loss_func)
    self.train_op = self.build_train_op()

    L2            = self.build_loss("L2")
    HUBER         = self.build_loss("HUBER")
    LMLS          = self.build_loss("LMLS")
    RELMSE        = self.build_loss("RELMSE")
    L1            = self.build_loss("L1")
    MAPE          = self.build_loss("MAPE")
    SSIM          = self.build_loss("SSIM")

    tf.summary.scalar("L2", L2)
    tf.summary.scalar("HUBER", HUBER)
    tf.summary.scalar("LMLS", LMLS)
    tf.summary.scalar("RELMSE", RELMSE)
    tf.summary.scalar("L1", L1)
    tf.summary.scalar("MAPE", MAPE)
    tf.summary.scalar("SSIM", SSIM)
    tf.summary.scalar("Learning_rate", self.lr)

  @abstractmethod  
  def build_model(self):
    pass
  
  def build_loss(self, loss_func):
    
    if loss_func == "L2":
      diff_square = tf.square(tf.subtract(self.outputs, self.refers))
      loss = tf.reduce_mean(diff_square)
      
    elif loss_func == 'HUBER':
      huber_loss = tf.losses.huber_loss(self.refers, self.outputs)
      loss = tf.reduce_mean(huber_loss)

    elif loss_func == "LMLS":
      diff = tf.subtract(self.outputs, self.refers)
      diff_square_ch_mean = tf.reduce_mean(tf.square(diff), axis=-1)
      loss = tf.reduce_mean(tf.log(1 + (0.5 * diff_square_ch_mean)))
    
    elif loss_func == "RELMSE":
      L2 = tf.square(tf.subtract(self.outputs, self.refers))
      denom = tf.square(self.refers) + 1.0e-2
      loss = tf.reduce_mean(L2 / denom)

    elif loss_func == "L1":
      diff = tf.abs(tf.subtract(self.outputs, self.refers))
      loss = tf.reduce_mean(diff)

    elif loss_func == 'MAPE':
      diff = tf.abs(tf.subtract(self.outputs, self.refers))
      diff = tf.div(diff, self.refers + 1.0+1e-2)
      loss = tf.reduce_mean(diff)

    elif loss_func == 'SSIM':
      loss = SSIM(self.outputs, self.refers)

    return loss
    

  def build_train_op(self):
    optim = tf.train.AdamOptimizer(self.lr)
    
    grads = optim.compute_gradients(self.loss)

    # Clip gradients to avoid exploding weights
    grads = [(None, var) if grad is None else (tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads]

    # Apply gradients
    apply_gradient_op = optim.apply_gradients(grads, global_step=self.global_step)
    with tf.control_dependencies([apply_gradient_op]):
      train_op = tf.no_op(name='Train')

    return train_op

def SSIM(a, b):

  # Generate filter kernel
  _, _, _, d = a.get_shape().as_list()
  window = generate_weight(5, 1.5)
  window = window / np.sum(window)
  window = window.astype(np.float32)
  window = window[:,:,np.newaxis,np.newaxis]
  window = tf.constant(window)
  window = tf.tile(window,[1, 1, d, 1])

  # Find means
  mA = tf.nn.depthwise_conv2d(a, window, strides=[1, 1, 1, 1], padding='VALID')
  mB = tf.nn.depthwise_conv2d(b, window, strides=[1, 1, 1, 1], padding='VALID')

  # Find standard deviations
  sA = tf.nn.depthwise_conv2d(a*a, window, strides=[1, 1, 1, 1], padding='VALID') - mA**2
  sB = tf.nn.depthwise_conv2d(b*b, window, strides=[1, 1, 1, 1], padding='VALID') - mB**2
  sAB = tf.nn.depthwise_conv2d(a*b, window, strides=[1, 1, 1, 1], padding='VALID') - mA*mB

  # Calc SSIM constants 
  L = 1.0
  k1 = 0.01
  k2 = 0.03
  c1 = (k1 * L)**2
  c2 = (k2 * L)**2

  # Plug into SSIM equation
  assert(c1 > 0 and c2 > 0)
  p1 = (2.0*mA*mB + c1)/(mA*mA + mB*mB + c1)
  p2 = (2.0*sAB + c2)/(sA + sB + c2)

  # We want to maximize SSIM or minimize (1-SSIM)
  return 1-tf.reduce_mean(p1*p2)

def generate_weight(radius, sigma):

  weight = np.zeros([2*radius + 1, 2*radius + 1])
  expFactor = -1.0 / (2 * sigma * sigma)
  for i in range(-radius, radius + 1):
    for j in range(-radius, radius + 1):
      weight[i+radius,j+radius] = np.exp(expFactor * (i * i + j * j))
  weightSum = np.sum(weight)
  assert(weightSum > 0)
  return weight / weightSum