import numpy as np
import tensorflow as tf
import os
from src.exr import write_exr, to_jpg
def decode_example(seralized_example, shape):
  features = tf.parse_single_example(seralized_example,
    features={
      'image/noisy_img'     : tf.FixedLenFeature([],
                                dtype=tf.string, default_value=''),
      'image/reference'     : tf.FixedLenFeature([],
                                dtype=tf.string, default_value=''),
    }
  )

  h, w, in_c, out_c = shape

  noisy_img      = tf.decode_raw(features['image/noisy_img'], tf.float32)
  reference      = tf.decode_raw(features['image/reference'], tf.float32)

  noisy_img      = tf.reshape(noisy_img, [h, w, in_c])
  reference      = tf.reshape(reference, [h, w, out_c])

  noisy_img_diffuse, noisy_img_specular,\
             reference_diffuse, reference_specular = preprocess(noisy_img, reference)

  return noisy_img_diffuse, noisy_img_specular, reference_diffuse, reference_specular

def next_batch_tensor(tfrecord_path, shape, batch_size=1,
                      shuffle_buffer=0, prefetch_size=1, repeat=0):
  ''' 다음 데이터를 출력하기 위한 텐서를 출력한다.
  Args:
    tfrecord_path  : 읽을 tfrecord 경로(---/---/파일이름.tfrecord)
    shape          : 높이, 폭, 입력 채널, 출력 채널의 시퀀스
                     ex) [65, 65, 66, 3] <-- h, w, in_c, out_c
    batch_size     : 미니배치 크기
    shuffle_buffer : 데이터 섞기 위한 버퍼 크기
    prefetch_size  : 모름. 그냥 1 씀
    repeat         : 데이터가 다 읽힌 경우 Exception이 발생한다.
                     이를 없애기 위해서는 몇 번 더 반복할지 정해줘야 한다.
  Returns:
    noisy_img      : noise가 있는 이미지 tensor
    reference      : noise가 없는 이미지 tensor(조금은 있겠지만..)
  '''

  dataset = tf.data.TFRecordDataset(tfrecord_path)
  dataset = dataset.map(lambda x: decode_example(x, shape))
  dataset = dataset.batch(batch_size)

  if shuffle_buffer > 0:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
  if prefetch_size > 0:
    dataset = dataset.prefetch(buffer_size=prefetch_size)
  if repeat > 0:
    dataset = dataset.repeat(repeat)

  iterator = dataset.make_one_shot_iterator()

  next_noise_image_diffuse, next_noise_image_specular,\
                 next_reference_diffuse, next_reference_specular = iterator.get_next()

  return next_noise_image_diffuse, next_noise_image_specular,\
                 next_reference_diffuse, next_reference_specular

def calc_grad(data):
  h, w, c = data.get_shape()

  dX = data[:, 1:, :] - data[:, :-1, :]
  dY = data[1:, :, :] - data[:-1, :, :]
  dX = tf.concat((tf.zeros([h, 1, c]), dX), axis=1)
  dY = tf.concat((tf.zeros([1, w, c]), dY), axis=0)

  return tf.concat((dX, dY), axis=-1)

def preprocess(noisy_img, reference):
  # =======================================
  color          = noisy_img[:, :, :3]
  color_v        = noisy_img[:, :, 3:4]
  specular       = noisy_img[:, :, 4:7]
  specular_v     = noisy_img[:, :, 7:8]
  diffuse        = noisy_img[:, :, 8:11]
  diffuse_v      = noisy_img[:, :, 11:12]
  normal         = noisy_img[:, :, 12:15]
  normal_v       = noisy_img[:, :, 15:16]
  albedo         = noisy_img[:, :, 16:19]
  albedo_v       = noisy_img[:, :, 19:20]
  depth          = noisy_img[:, :, 20:21]
  depth_v        = noisy_img[:, :, 21:22]

  color_v        = color_v    / (tf.square(tf.reduce_mean(color,   axis=-1, keepdims=True)) + 0.001)
  specular_v     = specular_v / (tf.square(tf.reduce_mean(specular,axis=-1, keepdims=True)) + 0.001)
  diffuse_v      = diffuse_v  / (tf.square(tf.reduce_mean(diffuse, axis=-1, keepdims=True)) + 0.001)
  normal_v       = normal_v   / (tf.square(tf.reduce_mean(normal,  axis=-1, keepdims=True)) + 0.001)
  albedo_v       = albedo_v   / (tf.square(tf.reduce_mean(albedo,  axis=-1, keepdims=True)) + 0.001)
  depth_v        = depth_v    / (tf.square(depth) + 0.001)

  color_grad    = calc_grad(color)
  specular_grad = calc_grad(specular)
  diffuse_grad  = calc_grad(diffuse)
  normal_grad   = calc_grad(normal)
  albedo_grad   = calc_grad(albedo)
  depth_grad    = calc_grad(depth)

  def median(image):
    image   = tf.expand_dims(image, axis=0)
    patches = tf.extract_image_patches(image, [1, 3, 3, 1], [1] * 4, [1] * 4, 'SAME')
    medians = tf.contrib.distributions.percentile(patches, 50.0, axis=3)
    return tf.squeeze(tf.transpose(medians, [0, 1, 2]), axis=0)

  noisy_img_diffuse = tf.concat(
                        [color, color_v, color_grad,
                         diffuse, diffuse_v, diffuse_grad,
                         normal, normal_v, normal_grad,
                         albedo, albedo_v, albedo_grad,
                         depth, depth_v, depth_grad], axis=-1)

  noisy_img_specular = tf.concat(
                        [color, color_v, color_grad,
                         specular, specular_v, specular_grad,
                         normal, normal_v, normal_grad,
                         albedo, albedo_v, albedo_grad,
                         depth, depth_v, depth_grad], axis=-1)

  reference_diffuse = reference[:, :, 6:9]
  reference_specular = reference[:, :, 3:6]

  ##########################################################################
  # for refer
  #
  # reference_diffuse = reference[:, :, 8:11]
  # reference_specular = reference[:, :, 4:7]
  ##########################################################################

  return noisy_img_diffuse, noisy_img_specular, reference_diffuse, reference_specular