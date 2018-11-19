import tensorflow as tf

def batch_norm(inputs, name=None):
  return tf.layers.batch_normalization(inputs, name=name)

def conv2d(inputs, filters, kernel_size=3, strides=1, 
           activation=None, padding='same', name=None):
  return tf.layers.conv2d(
    inputs             = inputs,
    filters            = filters,
    kernel_size        = kernel_size,
    strides            = strides,
    activation         = activation,
    padding            = padding,
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    name               = name
  )

def deconv2d(inputs, filters, kernel_size=3, strides=1, 
           activation=None, padding='same', name=None):
  return tf.layers.conv2d_transpose(
    inputs             = inputs,
    filters            = filters,
    kernel_size        = kernel_size,
    strides            = strides,
    activation         = activation,
    padding            = padding,
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    name               = name
  )