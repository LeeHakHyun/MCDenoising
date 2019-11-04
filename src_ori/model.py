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

    self.input_shape  = input_shape
    self.output_shape = output_shape

    self.inputs       = tf.placeholder(tf.float32, [None] + input_shape)
    self.refers       = tf.placeholder(tf.float32, [None] + output_shape)

    self.lr           = tf.train.exponential_decay(
                            start_lr,
                            tf.train.get_or_create_global_step(),
                            lr_decay_step,
                            lr_decay_rate,
                            staircase=True)

    self.outputs      = self.build_model()

    L2_loss           = self.build_loss('L2', self.outputs, self.refers)
    HUBER_loss        = self.build_loss('HUBER', self.outputs, self.refers)
    LMLS_loss         = self.build_loss('LMLS', self.outputs, self.refers)
    RELMSE_loss       = self.build_loss('RELMSE', self.outputs, self.refers)
    L1_loss           = self.build_loss('L1', self.outputs, self.refers)
    MAPE_loss         = self.build_loss('MAPE', self.outputs, self.refers)
    SSIM_loss         = self.build_loss('SSIM', self.outputs, self.refers)
    

    self.loss         = self.build_loss(loss_func, self.outputs, self.refers)

    self.train_op     = self.build_train_op(self.lr, self.loss)

    self.global_step  = tf.train.get_or_create_global_step()

    tf.summary.scalar('Loss', self.loss)
    tf.summary.scalar('Learning rate', self.lr)
    tf.summary.scalar('L2', L2_loss)
    tf.summary.scalar('HUBER', HUBER_loss)
    tf.summary.scalar('LMLS', LMLS_loss)
    tf.summary.scalar('RELMSE', RELMSE_loss)
    tf.summary.scalar('L1', L1_loss)
    tf.summary.scalar('MAPE', MAPE_loss)
    tf.summary.scalar('SSIM', SSIM_loss)

  def conv2d_module(self, inputs, feature, kernel, stride, activation):
    layer = conv2d(inputs, feature, kernel, stride, activation=activation)
    return tf.layers.batch_normalization(layer)

  def deconv2d_module(self, inputs, feature, kernel, stride, activation):
    layer = deconv2d(inputs, feature, kernel, stride, activation=activation)
    return tf.layers.batch_normalization(layer)

  def build_model(self):
    #######################################################################################
    #                             1x1
    
    # layer1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)
    # layer2 = self.conv2d_module(layer1, 64, 4, 2, tf.nn.relu)
    # layer3 = self.conv2d_module(layer2, 128, 4, 2, tf.nn.relu)
    # layer4 = self.conv2d_module(layer3, 256, 4, 2, tf.nn.relu)
    # layer5 = self.conv2d_module(layer4, 512, 4, 2, tf.nn.relu)

    # layer6 = self.deconv2d_module(layer5, 256, 4, 2, tf.nn.leaky_relu) + layer4
    # layer6 =               conv2d(layer6, 128, 1, 1, tf.nn.leaky_relu)
    # layer7 = self.deconv2d_module(layer6, 128, 4, 2, tf.nn.leaky_relu) + layer3
    # layer7 =               conv2d(layer7, 64, 1, 1, tf.nn.leaky_relu)
    # layer8 = self.deconv2d_module(layer7, 64, 4, 2, tf.nn.leaky_relu) + layer2
    # layer9 = self.deconv2d_module(layer8, 64, 4, 2, tf.nn.leaky_relu) + layer1

    # layer10 = self.conv2d_module(layer9,  64, 3, 1, tf.nn.leaky_relu)
    # layer10 =             conv2d(layer10, 32, 1, 1, tf.nn.leaky_relu)
    # layer11 = self.conv2d_module(layer10, 32, 3, 1, tf.nn.leaky_relu)
    # layer11 =             conv2d(layer11, 16, 1, 1, tf.nn.leaky_relu)
    # layer12 = self.conv2d_module(layer11, 16, 3, 1, tf.nn.leaky_relu)
    # layer13 = self.conv2d_module(layer12, 32, 3, 1, tf.nn.leaky_relu)
    # layer14 = self.conv2d_module(layer13, 64, 3, 1, tf.nn.leaky_relu)

    # layer = conv2d(layer14 + layer9, 3, 3, 1, activation=None)
    #######################################################################################

    #######################################################################################
    #                             MCDenoising
    #
    layer1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)

    layer2 = self.conv2d_module(layer1, 64, 4, 2, tf.nn.relu)
    layer3 = self.conv2d_module(layer2, 128, 4, 2, tf.nn.relu)
    layer4 = self.conv2d_module(layer3, 256, 4, 2, tf.nn.relu)

    layer5 = self.conv2d_module(layer4, 512, 4, 2, tf.nn.relu)

    layer6 = self.deconv2d_module(layer5, 256, 4, 2, tf.nn.leaky_relu) + layer4
    layer7 = self.deconv2d_module(layer6, 128, 4, 2, tf.nn.leaky_relu) + layer3
    layer8 = self.deconv2d_module(layer7, 64, 4, 2, tf.nn.leaky_relu) + layer2
    layer9 = self.deconv2d_module(layer8, 64, 4, 2, tf.nn.leaky_relu) + layer1

    layer10 = self.conv2d_module(layer9, 64, 3, 1, tf.nn.leaky_relu)
    layer11 = self.conv2d_module(layer10, 64, 3, 1, tf.nn.leaky_relu)
    layer12 = self.conv2d_module(layer11, 64, 3, 1, tf.nn.leaky_relu)
    layer13 = self.conv2d_module(layer12, 64, 3, 1, tf.nn.leaky_relu)
    layer14 = self.conv2d_module(layer13, 64, 3, 1, tf.nn.leaky_relu)

    layer = conv2d(layer14 + layer9, 3, 3, 1, activation=None)
    #######################################################################################

    #######################################################################################
    #                             RCNN
    #
    # layer1 = conv2d(self.inputs, 43, 3, 1, activation=tf.nn.relu)
    # layer2 = self.conv2d_module(layer1, 43, 3, 1, tf.nn.relu)
    # layer3 = self.conv2d_module(layer2, 43, 3, 1, tf.nn.relu)

    # layer4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(layer3)
    # layer5 = self.conv2d_module(layer4, 57, 3, 1, tf.nn.relu)
    # layer6 = self.conv2d_module(layer5, 57, 3, 1, tf.nn.relu)

    # layer7 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(layer6)
    # layer8 = self.conv2d_module(layer7, 76, 3, 1, tf.nn.relu)
    # layer9 = self.conv2d_module(layer8, 76, 3, 1, tf.nn.relu)

    # layer10 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(layer9)
    # layer11 = self.conv2d_module(layer10, 101, 3, 1, tf.nn.relu)
    # layer12 = self.conv2d_module(layer11, 101, 3, 1, tf.nn.relu)

    # layer13 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(layer12)
    # layer14 = self.conv2d_module(layer13, 101, 3, 1, tf.nn.relu)
    # layer15 = self.conv2d_module(layer14, 101, 3, 1, tf.nn.relu)    

    # layer16 = self.deconv2d_module(layer15, 101, 3, 2, tf.nn.leaky_relu) + layer12
    # layer17 = self.deconv2d_module(layer16, 76, 3, 2, tf.nn.leaky_relu) + layer9
    # layer18 = self.deconv2d_module(layer17, 57, 3, 2, tf.nn.leaky_relu) + layer6
    # layer19 = self.deconv2d_module(layer18, 43, 3, 2, tf.nn.leaky_relu) + layer3

    # layer = conv2d(layer19 + layer3, 3, 3, 1, activation=None)
    #######################################################################################


    return layer
