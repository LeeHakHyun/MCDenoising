import tensorflow as tf
from src.base_model import BaseModel
from src.tf_ops import conv2d, deconv2d

class DenoisingNet(BaseModel):
  def __init__(self, input_shape,
                     output_shape,
                     loss_func,
                     start_lr,
                     lr_decay_step,
                     lr_decay_rate):
  
    self.in_ch = input_shape[-1]

    super().__init__(input_shape,
                     output_shape,
                     loss_func,
                     start_lr,
                     lr_decay_step,
                     lr_decay_rate)

  def conv2d_module(self, inputs, feature, kernel, stride, activation):
    layer = conv2d(inputs, feature, kernel, stride, activation=activation)
    return tf.layers.batch_normalization(layer)
  
  def deconv2d_module(self, inputs, feature, kernel, stride, activation):
    layer = deconv2d(inputs, feature, kernel, stride, activation=activation)
    return tf.layers.batch_normalization(layer)
  
  def build_model(self):
    layer1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)

    layer2 = self.conv2d_module(layer1, 64, 4, 2, tf.nn.relu)
    layer3 = self.conv2d_module(layer2, 128, 4, 2, tf.nn.relu)
    layer4 = self.conv2d_module(layer3, 256, 4, 2, tf.nn.relu)
    
    layer5 = self.conv2d_module(layer4, 512, 4, 2, tf.nn.relu)
    
    layer6 = self.deconv2d_module(layer5, 256, 4, 2, tf.nn.leaky_relu) + layer4
    layer7 = self.deconv2d_module(layer6, 128, 4, 2, tf.nn.leaky_relu) + layer3
    layer8 = self.deconv2d_module(layer7, 64, 4, 2, tf.nn.leaky_relu) + layer2
    layer9 = self.deconv2d_module(layer8, 64, 4, 2, tf.nn.leaky_relu) + layer1

    layer10 = self.conv2d_module(layer9, 64, 3, 1, tf.nn.relu)
    layer11 = self.conv2d_module(layer10, 64, 3, 1, tf.nn.relu)
    layer12 = self.conv2d_module(layer11, 64, 3, 1, tf.nn.relu)
    layer13 = self.conv2d_module(layer12, 64, 3, 1, tf.nn.relu)
    layer14 = self.conv2d_module(layer13, 64, 3, 1, tf.nn.relu)
    
    layer = conv2d(layer14 + layer9, 3, 3, 1, activation=None)

    return layer



