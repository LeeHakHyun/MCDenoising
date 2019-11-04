import tensorflow as tf
from src.base_model import BaseModel
from src.tf_ops import conv2d, deconv2d

class DenoisingNet(BaseModel):
  def __init__(self, input_shape,
                     output_shape,
                     loss_func,
                     start_lr,
                     lr_decay_step,
                     lr_decay_rate,
                     diff_spec_select):

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

    self.outputs      = self.build_model(diff_spec_select)

    self.loss         = self.build_loss(loss_func, self.outputs, self.refers)

    self.train_op     = self.build_train_op(self.lr, self.loss)

    self.global_step  = tf.train.get_or_create_global_step()

  def conv2d_module(self, inputs, feature, kernel, stride, activation):
    layer = conv2d  (inputs, feature, kernel, stride, activation=tf.nn.relu)
    return tf.layers.batch_normalization(layer)

  def deconv2d_module(self, inputs, feature, kernel, stride, activation):
    layer = deconv2d(inputs, feature, kernel, stride, activation=activation)
    return tf.layers.batch_normalization(layer)
  # ===============================================================================================================================
  # 4
  #
  def build_model(self, diff_spec_select):
    if(diff_spec_select == "Diffuse"):
        layer1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)
        layer2 = self.conv2d_module(layer1, 64, 4, 2, tf.nn.relu)
        layer3 = self.conv2d_module(layer2, 128, 4, 2, tf.nn.relu)
        layer4 = self.conv2d_module(layer3, 256, 4, 2, tf.nn.relu)
        layer5 = self.conv2d_module(layer4, 512, 4, 2, tf.nn.relu)
        layer5 =               conv2d(layer5, 256, 1, 1, tf.nn.relu)
        layer6 = self.deconv2d_module(layer5, 256, 4, 2, tf.nn.leaky_relu) + layer4
        layer6 =               conv2d(layer6, 128, 1, 1, tf.nn.leaky_relu)
        layer7 = self.deconv2d_module(layer6, 128, 4, 2, tf.nn.leaky_relu) + layer3
        layer7 =               conv2d(layer7, 64, 1, 1, tf.nn.leaky_relu)
        layer8 = self.deconv2d_module(layer7, 64, 4, 2, tf.nn.leaky_relu) + layer2
        layer9 = self.deconv2d_module(layer8, 64, 4, 2, tf.nn.leaky_relu) + layer1
        layer10 = self.conv2d_module(layer9,  64, 3, 1, tf.nn.leaky_relu)
        layer10 =             conv2d(layer10, 32, 1, 1, tf.nn.leaky_relu)
        layer11 = self.conv2d_module(layer10, 32, 3, 1, tf.nn.leaky_relu)
        layer11 =             conv2d(layer11, 16, 1, 1, tf.nn.leaky_relu)
        layer12 = self.conv2d_module(layer11, 16, 3, 1, tf.nn.leaky_relu)
        layer13 = self.conv2d_module(layer12, 32, 3, 1, tf.nn.leaky_relu)
        layer14 = self.conv2d_module(layer13, 64, 3, 1, tf.nn.leaky_relu)

        layer = conv2d(layer14 + layer9, 3, 3, 1, activation=None)
        return layer

    elif(diff_spec_select == "Specular"):
        layer1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)
        layer2 = self.conv2d_module(layer1, 64, 4, 2, tf.nn.relu)
        layer3 = self.conv2d_module(layer2, 128, 4, 2, tf.nn.relu)
        layer4 = self.conv2d_module(layer3, 256, 4, 2, tf.nn.relu)
        layer5 = self.conv2d_module(layer4, 512, 4, 2, tf.nn.relu)
        layer5 =               conv2d(layer5, 256, 1, 1, tf.nn.relu)
        layer6 = self.deconv2d_module(layer5, 256, 4, 2, tf.nn.leaky_relu) + layer4
        layer6 =               conv2d(layer6, 128, 1, 1, tf.nn.leaky_relu)
        layer7 = self.deconv2d_module(layer6, 128, 4, 2, tf.nn.leaky_relu) + layer3
        layer7 =               conv2d(layer7, 64, 1, 1, tf.nn.leaky_relu)
        layer8 = self.deconv2d_module(layer7, 64, 4, 2, tf.nn.leaky_relu) + layer2
        layer9 = self.deconv2d_module(layer8, 64, 4, 2, tf.nn.leaky_relu) + layer1
        layer10 = self.conv2d_module(layer9,  64, 3, 1, tf.nn.leaky_relu)
        layer10 =             conv2d(layer10, 32, 1, 1, tf.nn.leaky_relu)
        layer11 = self.conv2d_module(layer10, 32, 3, 1, tf.nn.leaky_relu)
        layer11 =             conv2d(layer11, 16, 1, 1, tf.nn.leaky_relu)
        layer12 = self.conv2d_module(layer11, 16, 3, 1, tf.nn.leaky_relu)
        layer13 = self.conv2d_module(layer12, 32, 3, 1, tf.nn.leaky_relu)
        layer14 = self.conv2d_module(layer13, 64, 3, 1, tf.nn.leaky_relu)

        layer = conv2d(layer14 + layer9, 3, 3, 1, activation=None)
        return layer
  # ===============================================================================================================================

  # ===============================================================================================================================
  # 5
  #
  # def build_model(self, diff_spec_select):
  #   if(diff_spec_select == "Diffuse"):
  #       layer1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)
  #       layer2 = self.conv2d_module(layer1, 64, 4, 2, tf.nn.relu)
  #       layer3 = self.conv2d_module(layer2, 128, 4, 2, tf.nn.relu)
  #       layer4 = self.conv2d_module(layer3, 256, 4, 2, tf.nn.relu)
  #       layer5 = self.conv2d_module(layer4, 512, 4, 2, tf.nn.relu)
  #       #layer5 =               conv2d(layer5, 256, 1, 1, tf.nn.relu)
  #       layer6 = self.deconv2d_module(layer5, 256, 4, 2, tf.nn.leaky_relu) + layer4
  #       #layer6 =               conv2d(layer6, 128, 1, 1, tf.nn.leaky_relu)
  #       layer7 = self.deconv2d_module(layer6, 128, 4, 2, tf.nn.leaky_relu) + layer3
  #       #layer7 =               conv2d(layer7, 64, 1, 1, tf.nn.leaky_relu)
  #       layer8 = self.deconv2d_module(layer7, 64, 4, 2, tf.nn.leaky_relu) + layer2
  #       layer9 = self.deconv2d_module(layer8, 64, 4, 2, tf.nn.leaky_relu) + layer1

  #       layer10 = self.conv2d_module(layer9,  64, 3, 1, tf.nn.leaky_relu)
  #       #layer10 =             conv2d(layer10, 32, 1, 1, tf.nn.leaky_relu)
  #       layer11 = self.conv2d_module(layer10, 32, 3, 1, tf.nn.leaky_relu)
  #       #layer11 =             conv2d(layer11, 16, 1, 1, tf.nn.leaky_relu)
  #       layer12 = self.conv2d_module(layer11, 16, 3, 1, tf.nn.leaky_relu)
  #       layer13 = self.conv2d_module(layer12, 32, 3, 1, tf.nn.leaky_relu)
  #       layer14 = self.conv2d_module(layer13, 64, 3, 1, tf.nn.leaky_relu)


  #       layer = conv2d(layer14 + layer9, 3, 3, 1, activation=None)
  #       return layer

  #   elif(diff_spec_select == "Specular"):
  #       layer1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)
  #       layer2 = self.conv2d_module(layer1, 64, 4, 2, tf.nn.relu)
  #       layer3 = self.conv2d_module(layer2, 128, 4, 2, tf.nn.relu)
  #       layer4 = self.conv2d_module(layer3, 256, 4, 2, tf.nn.relu)
  #       layer5 = self.conv2d_module(layer4, 512, 4, 2, tf.nn.relu)
  #       #layer5 =               conv2d(layer5, 256, 1, 1, tf.nn.relu)
  #       layer6 = self.deconv2d_module(layer5, 256, 4, 2, tf.nn.leaky_relu) + layer4
  #       #layer6 =               conv2d(layer6, 128, 1, 1, tf.nn.leaky_relu)
  #       layer7 = self.deconv2d_module(layer6, 128, 4, 2, tf.nn.leaky_relu) + layer3
  #       #layer7 =               conv2d(layer7, 64, 1, 1, tf.nn.leaky_relu)
  #       layer8 = self.deconv2d_module(layer7, 64, 4, 2, tf.nn.leaky_relu) + layer2
  #       layer9 = self.deconv2d_module(layer8, 64, 4, 2, tf.nn.leaky_relu) + layer1

  #       layer10 = self.conv2d_module(layer9,  64, 3, 1, tf.nn.leaky_relu)
  #       #layer10 =             conv2d(layer10, 32, 1, 1, tf.nn.leaky_relu)
  #       layer11 = self.conv2d_module(layer10, 32, 3, 1, tf.nn.leaky_relu)
  #       #layer11 =             conv2d(layer11, 16, 1, 1, tf.nn.leaky_relu)
  #       layer12 = self.conv2d_module(layer11, 16, 3, 1, tf.nn.leaky_relu)
  #       layer13 = self.conv2d_module(layer12, 32, 3, 1, tf.nn.leaky_relu)
  #       layer14 = self.conv2d_module(layer13, 64, 3, 1, tf.nn.leaky_relu)

  #       layer = conv2d(layer14 + layer9, 3, 3, 1, activation=None)
  #       return layer
  # ===============================================================================================================================


  # ===============================================================================================================================
  # 3
  # def build_model(self,diff_spec_select):
  #   if(diff_spec_select == "Diffuse"):
  #       layer_diff1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)
  #       layer_diff2  = self.conv2d_module   (layer_diff1,  64,  5, 2, tf.nn.relu)
  #       layer_diff3  = self.conv2d_module   (layer_diff2,  128, 5, 2, tf.nn.relu)
  #       layer_diff4  = self.conv2d_module   (layer_diff3,  256, 3, 2, tf.nn.relu)
  #       layer_diff5  = self.conv2d_module   (layer_diff4,  512, 3, 2, tf.nn.relu)
  #       layer_diff5  = conv2d               (layer_diff5,  256, 1, 1, tf.nn.leaky_relu)        
  #       layer_diff6  = self.deconv2d_module (layer_diff5,  256, 3, 2, tf.nn.leaky_relu) + layer_diff4
  #       layer_diff6  = conv2d               (layer_diff6,  128, 1, 1, tf.nn.leaky_relu)   
  #       layer_diff7  = self.deconv2d_module (layer_diff6,  128, 5, 2, tf.nn.leaky_relu) + layer_diff3
  #       layer_diff7  = conv2d               (layer_diff7,  64,  1, 1, tf.nn.leaky_relu) 
  #       layer_diff8  = self.deconv2d_module (layer_diff7,  64,  5, 2, tf.nn.leaky_relu) + layer_diff2   
  #       layer_diff9  = self.deconv2d_module (layer_diff8,  64,  5, 2, tf.nn.leaky_relu) + layer_diff1
  #       layer_diff11 = self.conv2d_module   (layer_diff9,  64,  3, 1, tf.nn.leaky_relu)
  #       layer_diff11 = conv2d               (layer_diff11, 32,  1, 1, tf.nn.leaky_relu)        
  #       layer_diff12 = self.conv2d_module   (layer_diff11, 32,  3, 1, tf.nn.leaky_relu)
  #       layer_diff12 = conv2d               (layer_diff12, 16,  1, 1, tf.nn.leaky_relu)
  #       layer_diff13 = self.conv2d_module   (layer_diff12, 16,  3, 1, tf.nn.leaky_relu)
  #       layer_diff14 = conv2d               (layer_diff13, 8,   1, 1, tf.nn.leaky_relu)
  #       layer_diff14 = self.conv2d_module   (layer_diff13, 8,   3, 1, tf.nn.leaky_relu)
  #       layer_diff = conv2d(layer_diff14, 3, 3, 1, activation=None)
  #       return layer_diff

  #   elif(diff_spec_select == "Specular"):
  #       layer_spec1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)
  #       layer_spec1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)
  #       layer_spec2  = self.conv2d_module   (layer_spec1,  64,  5, 2, tf.nn.relu)
  #       layer_spec3  = self.conv2d_module   (layer_spec2,  128, 5, 2, tf.nn.relu)
  #       layer_spec4  = self.conv2d_module   (layer_spec3,  256, 3, 2, tf.nn.relu)
  #       layer_spec5  = self.conv2d_module   (layer_spec4,  512, 3, 2, tf.nn.relu)
  #       layer_spec5  = conv2d               (layer_spec5,  256, 1, 1, tf.nn.leaky_relu)        
  #       layer_spec6  = self.deconv2d_module (layer_spec5,  256, 3, 2, tf.nn.leaky_relu) + layer_spec4
  #       layer_spec6  = conv2d               (layer_spec6,  128, 1, 1, tf.nn.leaky_relu)   
  #       layer_spec7  = self.deconv2d_module (layer_spec6,  128, 5, 2, tf.nn.leaky_relu) + layer_spec3
  #       layer_spec7  = conv2d               (layer_spec7,  64,  1, 1, tf.nn.leaky_relu) 
  #       layer_spec8  = self.deconv2d_module (layer_spec7,  64,  5, 2, tf.nn.leaky_relu) + layer_spec2
  #       layer_spec9  = self.deconv2d_module (layer_spec8,  64,  5, 2, tf.nn.leaky_relu) + layer_spec1
  #       layer_spec11 = self.conv2d_module   (layer_spec9,  64,  3, 1, tf.nn.leaky_relu)
  #       layer_spec11 = conv2d               (layer_spec11, 32,  1, 1, tf.nn.leaky_relu)        
  #       layer_spec12 = self.conv2d_module   (layer_spec11, 32,  3, 1, tf.nn.leaky_relu)
  #       layer_spec12 = conv2d               (layer_spec12, 16,  1, 1, tf.nn.leaky_relu)
  #       layer_spec13 = self.conv2d_module   (layer_spec12, 16,  3, 1, tf.nn.leaky_relu) 
  #       layer_spec14 = conv2d               (layer_spec13, 8,   1, 1, tf.nn.leaky_relu)
  #       layer_spec14 = self.conv2d_module   (layer_spec13, 8,   3, 1, tf.nn.leaky_relu)
  #       layer_spec = conv2d(layer_spec14, 3, 3, 1, activation=None)
  #       return layer_spec
  # ===============================================================================================================================

  # ===============================================================================================================================
  # 1
  # def build_model(self,diff_spec_select):
  #   if(diff_spec_select == "Diffuse"):
  #       layer_diff1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)
  #       layer_diff2  = self.conv2d_module   (layer_diff1,  64,  5, 2, tf.nn.relu)
  #       layer_diff3  = self.conv2d_module   (layer_diff2,  128, 5, 2, tf.nn.relu)
  #       layer_diff4  = self.conv2d_module   (layer_diff3,  256, 3, 2, tf.nn.relu)
  #       layer_diff5  = self.conv2d_module   (layer_diff4,  512, 3, 2, tf.nn.relu)
  #       #layer_diff5  = conv2d               (layer_diff5,  256, 1, 1, tf.nn.leaky_relu)        
  #       layer_diff6  = self.deconv2d_module (layer_diff5,  256, 3, 2, tf.nn.leaky_relu) + layer_diff4
  #       #layer_diff6  = conv2d               (layer_diff6,  128, 1, 1, tf.nn.leaky_relu)   
  #       layer_diff7  = self.deconv2d_module (layer_diff6,  128, 5, 2, tf.nn.leaky_relu) + layer_diff3
  #       #layer_diff7  = conv2d               (layer_diff7,  64,  1, 1, tf.nn.leaky_relu) 
  #       layer_diff8  = self.deconv2d_module (layer_diff7,  64,  5, 2, tf.nn.leaky_relu) + layer_diff2   
  #       layer_diff9  = self.deconv2d_module (layer_diff8,  64,  5, 2, tf.nn.leaky_relu) + layer_diff1
  #       layer_diff11 = self.conv2d_module   (layer_diff9,  64,  3, 1, tf.nn.leaky_relu)
  #       #layer_diff11 = conv2d               (layer_diff11, 32,  1, 1, tf.nn.leaky_relu)        
  #       layer_diff12 = self.conv2d_module   (layer_diff11, 32,  3, 1, tf.nn.leaky_relu)
  #       #layer_diff12 = conv2d               (layer_diff12, 16,  1, 1, tf.nn.leaky_relu)
  #       layer_diff13 = self.conv2d_module   (layer_diff12, 16,  3, 1, tf.nn.leaky_relu)
  #       #layer_diff14 = conv2d               (layer_diff13, 8,   1, 1, tf.nn.leaky_relu)
  #       layer_diff14 = self.conv2d_module   (layer_diff13, 8,   3, 1, tf.nn.leaky_relu)
  #       layer_diff = conv2d(layer_diff14, 3, 3, 1, activation=None)
  #       return layer_diff

  #   elif(diff_spec_select == "Specular"):
  #       layer_spec1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)
  #       layer_spec1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)
  #       layer_spec2  = self.conv2d_module   (layer_spec1,  64,  5, 2, tf.nn.relu)
  #       layer_spec3  = self.conv2d_module   (layer_spec2,  128, 5, 2, tf.nn.relu)
  #       layer_spec4  = self.conv2d_module   (layer_spec3,  256, 3, 2, tf.nn.relu)
  #       layer_spec5  = self.conv2d_module   (layer_spec4,  512, 3, 2, tf.nn.relu)
  #       #layer_spec5  = conv2d               (layer_spec5,  256, 1, 1, tf.nn.leaky_relu)        
  #       layer_spec6  = self.deconv2d_module (layer_spec5,  256, 3, 2, tf.nn.leaky_relu) + layer_spec4
  #       #layer_spec6  = conv2d               (layer_spec6,  128, 1, 1, tf.nn.leaky_relu)   
  #       layer_spec7  = self.deconv2d_module (layer_spec6,  128, 5, 2, tf.nn.leaky_relu) + layer_spec3
  #       #layer_spec7  = conv2d               (layer_spec7,  64,  1, 1, tf.nn.leaky_relu) 
  #       layer_spec8  = self.deconv2d_module (layer_spec7,  64,  5, 2, tf.nn.leaky_relu) + layer_spec2
  #       layer_spec9  = self.deconv2d_module (layer_spec8,  64,  5, 2, tf.nn.leaky_relu) + layer_spec1
  #       layer_spec11 = self.conv2d_module   (layer_spec9,  64,  3, 1, tf.nn.leaky_relu)
  #       #layer_spec11 = conv2d               (layer_spec11, 32,  1, 1, tf.nn.leaky_relu)        
  #       layer_spec12 = self.conv2d_module   (layer_spec11, 32,  3, 1, tf.nn.leaky_relu)
  #       #layer_spec12 = conv2d               (layer_spec12, 16,  1, 1, tf.nn.leaky_relu)
  #       layer_spec13 = self.conv2d_module   (layer_spec12, 16,  3, 1, tf.nn.leaky_relu) 
  #       #layer_spec14 = conv2d               (layer_spec13, 8,   1, 1, tf.nn.leaky_relu)
  #       layer_spec14 = self.conv2d_module   (layer_spec13, 8,   3, 1, tf.nn.leaky_relu)
  #       layer_spec = conv2d(layer_spec14, 3, 3, 1, activation=None)
  #       return layer_spec
  # ===============================================================================================================================

  # ===============================================================================================================================

  # def build_model(self,diff_spec_select):
  #   if(diff_spec_select == "Diffuse"):
  #       layer_diff1 = conv2d(self.inputs, 64, 11, 1, activation=tf.nn.relu)

  #       layer_diff2  = self.conv2d_module   (layer_diff1,  64,  7, 2, tf.nn.relu)
  #       layer_diff3  = self.conv2d_module   (layer_diff2,  128, 5, 2, tf.nn.relu)
  #       layer_diff4  = self.conv2d_module   (layer_diff3,  256, 3, 2, tf.nn.relu)
  #       layer_diff5  = self.conv2d_module   (layer_diff4,  512, 3, 2, tf.nn.relu)
  #       layer_diff6  = self.deconv2d_module (layer_diff5,  256, 3, 2, tf.nn.leaky_relu) + layer_diff4
  #       layer_diff7  = self.deconv2d_module (layer_diff6,  128, 5, 2, tf.nn.leaky_relu) + layer_diff3
  #       layer_diff8  = self.deconv2d_module (layer_diff7,  64,  5, 2, tf.nn.leaky_relu)
  #       layer_diff9  = self.deconv2d_module (layer_diff8,  64,  7, 2, tf.nn.leaky_relu)
        
  #       layer_diff10 = self.conv2d_module   (layer_diff9,  64,  3, 1, tf.nn.leaky_relu)
  #       layer_diff11 = self.conv2d_module   (layer_diff10, 64,  3, 1, tf.nn.leaky_relu)
  #       layer_diff12 = self.conv2d_module   (layer_diff11, 32,  3, 1, tf.nn.leaky_relu)
  #       layer_diff13 = self.conv2d_module   (layer_diff12, 16,  3, 1, tf.nn.leaky_relu)
  #       layer_diff14 = self.conv2d_module   (layer_diff13, 8,   3, 1, tf.nn.leaky_relu)

  #       layer_diff = conv2d(layer_diff14, 3, 3, 1, activation=None)

  #       return layer_diff

  #   elif(diff_spec_select == "Specular"):
  #       layer_spec1 = conv2d(self.inputs, 64, 11, 1, activation=tf.nn.relu)

  #       layer_spec2  = self.conv2d_module   (layer_spec1,  64,  7, 2, tf.nn.relu)
  #       layer_spec3  = self.conv2d_module   (layer_spec2,  128, 5, 2, tf.nn.relu)
  #       layer_spec4  = self.conv2d_module   (layer_spec3,  256, 3, 2, tf.nn.relu)
  #       layer_spec5  = self.conv2d_module   (layer_spec4,  512, 3, 2, tf.nn.relu)
  #       layer_spec6  = self.deconv2d_module (layer_spec5,  256, 3, 2, tf.nn.leaky_relu) + layer_spec4
  #       layer_spec7  = self.deconv2d_module (layer_spec6,  128, 5, 2, tf.nn.leaky_relu) + layer_spec3
  #       layer_spec8  = self.deconv2d_module (layer_spec7,  64,  5, 2, tf.nn.leaky_relu)
  #       layer_spec9  = self.deconv2d_module (layer_spec8,  64,  7, 2, tf.nn.leaky_relu)

  #       layer_spec10 = self.conv2d_module   (layer_spec9,  64,  3, 1, tf.nn.leaky_relu)
  #       layer_spec11 = self.conv2d_module   (layer_spec10, 64,  3, 1, tf.nn.leaky_relu)
  #       layer_spec12 = self.conv2d_module   (layer_spec11, 32,  3, 1, tf.nn.leaky_relu)
  #       layer_spec13 = self.conv2d_module   (layer_spec12, 16,  3, 1, tf.nn.leaky_relu)
  #       layer_spec14 = self.conv2d_module   (layer_spec13, 8,   3, 1, tf.nn.leaky_relu)

  #       layer_spec = conv2d(layer_spec14, 3, 3, 1, activation=None)

  #       return layer_spec

  # def build_model(self,diff_spec_select):
  #   if(diff_spec_select == "Diffuse"):
  #       layer_diff1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)

  #       layer_diff2 = self.conv2d_module(layer_diff1, 64, 4, 2, tf.nn.relu)
  #       layer_diff3 = self.conv2d_module(layer_diff2, 128, 4, 2, tf.nn.relu)
  #       layer_diff4 = self.conv2d_module(layer_diff3, 256, 4, 2, tf.nn.relu)
  #       layer_diff5 = self.conv2d_module(layer_diff4, 512, 4, 2, tf.nn.relu)
  #       layer_diff6 = self.deconv2d_module(layer_diff5, 256, 4, 2, tf.nn.leaky_relu) + layer_diff4
  #       layer_diff7 = self.deconv2d_module(layer_diff6, 128, 4, 2, tf.nn.leaky_relu) + layer_diff3
  #       layer_diff8 = self.deconv2d_module(layer_diff7, 64, 4, 2, tf.nn.leaky_relu) + layer_diff2
  #       layer_diff9 = self.deconv2d_module(layer_diff8, 64, 4, 2, tf.nn.leaky_relu) + layer_diff1
  #       layer_diff10 = self.conv2d_module(layer_diff9, 64, 3, 1, tf.nn.leaky_relu)
  #       layer_diff11 = self.conv2d_module(layer_diff10, 64, 3, 1, tf.nn.leaky_relu)
  #       layer_diff12 = self.conv2d_module(layer_diff11, 64, 3, 1, tf.nn.leaky_relu)
  #       layer_diff13 = self.conv2d_module(layer_diff12, 64, 3, 1, tf.nn.leaky_relu)
  #       layer_diff14 = self.conv2d_module(layer_diff13, 64, 3, 1, tf.nn.leaky_relu)

  #       layer_diff = conv2d(layer_diff14 + layer_diff9, 3, 3, 1, activation=None)

  #       return layer_diff

  #   elif(diff_spec_select == "Specular"):
  #       layer_spec1 = conv2d(self.inputs, 64, 7, 1, activation=tf.nn.relu)

  #       layer_spec2 = self.conv2d_module(layer_spec1, 64, 4, 2, tf.nn.relu)
  #       layer_spec3 = self.conv2d_module(layer_spec2, 128, 4, 2, tf.nn.relu)
  #       layer_spec4 = self.conv2d_module(layer_spec3, 256, 4, 2, tf.nn.relu)
  #       layer_spec5 = self.conv2d_module(layer_spec4, 512, 4, 2, tf.nn.relu)
  #       layer_spec6 = self.deconv2d_module(layer_spec5, 256, 4, 2, tf.nn.leaky_relu) + layer_spec4
  #       layer_spec7 = self.deconv2d_module(layer_spec6, 128, 4, 2, tf.nn.leaky_relu) + layer_spec3
  #       layer_spec8 = self.deconv2d_module(layer_spec7, 64, 4, 2, tf.nn.leaky_relu) + layer_spec2
  #       layer_spec9 = self.deconv2d_module(layer_spec8, 64, 4, 2, tf.nn.leaky_relu) + layer_spec1
  #       layer_spec10 = self.conv2d_module(layer_spec9, 64, 3, 1, tf.nn.leaky_relu)
  #       layer_spec11 = self.conv2d_module(layer_spec10, 64, 3, 1, tf.nn.leaky_relu)
  #       layer_spec12 = self.conv2d_module(layer_spec11, 64, 3, 1, tf.nn.leaky_relu)
  #       layer_spec13 = self.conv2d_module(layer_spec12, 64, 3, 1, tf.nn.leaky_relu)
  #       layer_spec14 = self.conv2d_module(layer_spec13, 64, 3, 1, tf.nn.leaky_relu)

  #       layer_spec = conv2d(layer_spec14 + layer_spec9, 3, 3, 1, activation=None)

  #       return layer_spec
  # ===============================================================================================================================
