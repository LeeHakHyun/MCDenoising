import os
import random
import tensorflow as tf
import cv2
import src.utils as utils
import shutil
import numpy as np
import time
#from PIL import Image
from itertools import count
from src_ori.model import DenoisingNet
from src_ori.dataset import next_batch_tensor
from src_ori.base_model import SSIM
from src_ori.exr import write_exr, to_jpg

BATCH_SIZE        = 32
IMG_HEIGHT        = 720
IMG_WIDTH         = 1280
PATCH_SIZE        = 64
N_NOISY_FEATURE   = 22
N_REFER_FEATURE   = 9
INPUT_CH          = 63
OUTPUT_CH         = 3

VALID_TFRECORD    = './data/valid.tfrecord'

DEBUG_DIR         = './debug/'

DEBUG_IMAGE_DIR   = DEBUG_DIR + 'images/1x1_demo/'
NOISY_IMAGE_DIR   = os.path.join(DEBUG_IMAGE_DIR, "noisy_img/")
REFER_IMAGE_DIR   = os.path.join(DEBUG_IMAGE_DIR, "reference/")
DENOISED_IMG_DIR  = os.path.join(DEBUG_IMAGE_DIR, "denoised_img/")

CKPT_DIR          = DEBUG_DIR + 'checkpoint/1x1_2/'

net = DenoisingNet(input_shape=[None, None, INPUT_CH],
                     output_shape=[None, None, OUTPUT_CH],
                     loss_func='RELMSE',
                     start_lr=0.0001,
                     lr_decay_step=10000,
                     lr_decay_rate=1.0)

sess = tf.Session()

saver = tf.train.Saver()

print('Saver initialized')
recent_ckpt_job_path = tf.train.latest_checkpoint(CKPT_DIR)

if recent_ckpt_job_path is None:
  raise FileNotFoundError("There is no files in", CKPT_DIR)
else:
  saver.restore(sess, recent_ckpt_job_path)
  print("Restoring...", recent_ckpt_job_path)

valid_noisy_img, valid_reference = \
  next_batch_tensor(
    tfrecord_path = VALID_TFRECORD,
    shape = [IMG_HEIGHT, IMG_WIDTH, N_NOISY_FEATURE, N_REFER_FEATURE]
  )

loss_arr = []
loss_SSIM_arr = []
for step in count():
  # 더이상 읽을 것이 없으면 빠져 나오고 다음 epoch으로
  try:
    noisy_img, reference = sess.run([valid_noisy_img, valid_reference])
  except tf.errors.OutOfRangeError as e:
    print("done")
    break
    step = 0
    valid_noisy_img, valid_reference = \
    next_batch_tensor(
      tfrecord_path = VALID_TFRECORD,
      shape = [IMG_HEIGHT, IMG_WIDTH, N_NOISY_FEATURE, N_REFER_FEATURE]
    )
    break
  T = time.time()
  loss, denoised_img = sess.run([net.loss, net.outputs],
                                          feed_dict={net.inputs:noisy_img,
                                                      net.refers:reference})
  print(f'inference time = {time.time() - T:.5f}', end=" ")


#  L2 = tf.square(tf.subtract(valid_noisy_img[0, 150:250, 570:670, :3] , reference[0, 150:250, 570:670, :3] ))
#  denom = tf.square(reference[0, 150:250, 570:670, :3]) + 1.0e-2
#  loss = sess.run(tf.reduce_mean(L2 / denom))
#  loss_SSIM = sess.run(SSIM(valid_noisy_img[0, 150:250, 570:670, :3], reference[0, 150:250, 570:670, :3]))

  L2 = tf.square(tf.subtract(denoised_img, reference))
  denom = tf.square(reference) + 1.0e-2
  loss = sess.run(tf.reduce_mean(L2 / denom))
  loss_SSIM = sess.run(SSIM(denoised_img, reference))

  loss_arr.append(loss);
  loss_SSIM_arr.append(loss_SSIM);
  rp = REFER_IMAGE_DIR + f'{step+1}'
  dp = DENOISED_IMG_DIR + f'{step+1}'
  np = NOISY_IMAGE_DIR + f'{step+1}'
  print(f"{step:2d} RELMSE loss = {loss:.4f}  SSIM loss = {loss_SSIM:.4f}");


  write_exr(reference[0, :, :, :3], rp + ".exr")
#  denoised_img[0, 144:149, 564:676, 0] = 0; denoised_img[0, 144:149, 564:676, 1] = 0;   denoised_img[0, 144:149, 564:676, 2] = 255;
#  denoised_img[0, 251:256, 564:676, 0] = 0; denoised_img[0, 251:256, 564:676, 1] = 0;   denoised_img[0, 251:256, 564:676, 2] = 255;
#  denoised_img[0, 144:256, 564:569, 0] = 0; denoised_img[0, 144:256, 564:569, 1] = 0;   denoised_img[0, 144:256, 564:569, 2] = 255;
#  denoised_img[0, 144:256, 671:676, 0] = 0; denoised_img[0, 144:256, 671:676, 1] = 0;   denoised_img[0, 144:256, 671:676, 2] = 255;

  write_exr(denoised_img[0, :, :, :3], dp + ".exr")
  write_exr(noisy_img[0, :, :,:3], np + ".exr")

  write_exr(reference[0, 150:250, 570:670, :3] , rp + "_small" + ".exr")
  write_exr(denoised_img[0, 150:250, 570:670, :3] , dp + "_small_RELMSE_" + f"{loss:.4f}" + "SSIM_" + f"{loss_SSIM:.4f}" + ".exr")
  write_exr(noisy_img[0, 150:250, 570:670, :3] , np + "_small" + ".exr")

  to_jpg(rp + ".exr", rp + ".jpg")
  to_jpg(dp + ".exr", dp + ".jpg")
  to_jpg(np + ".exr", np + ".jpg")

  to_jpg(rp + "_small.exr", rp + "_small.jpg")
  to_jpg(dp + "_small_RELMSE_" + f"{loss:.4f}" + "SSIM_" + f"{loss_SSIM:.4f}" + ".exr", dp + "_small_RELMSE_" + f"{loss:.4f}" + "SSIM_" + f"{loss_SSIM:.4f}" + ".jpg")
  to_jpg(np + "_small.exr", np + "_small.jpg")

  os.remove(rp + "_small" + ".exr")
  os.remove(dp + "_small_RELMSE_" + f"{loss:.4f}" + "SSIM_" + f"{loss_SSIM:.4f}" + ".exr")
  os.remove(np + "_small" + ".exr")

  os.remove(rp + ".exr")
  os.remove(dp + ".exr")
  os.remove(np + ".exr")

print("Mean loss RelMSE = ",sess.run(tf.reduce_mean(loss_arr)));
print("Mean loss SSIM = ", sess.run(tf.reduce_mean(loss_SSIM_arr)));


"""
  b, h, w, c = noisy_img.shape

  W = w//3

  compare_image = np.hstack((noisy_img[0, :, :W, :3],
                             denoised_img[0, :, W:W*2, :3],
                             reference[0, :, W*2:, :3]))
  cv2.line(compare_image, (W, 0), (W, h), (50, 50, 50), 3)
  cv2.line(compare_image, (W*2, 0), (W*2, h), (50, 50, 50), 3)

  cv2.putText(compare_image, 'Noisy Image', (W//4, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
  cv2.putText(compare_image, '(128 spp)', (W//4, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

  cv2.putText(compare_image, 'Reference', (W//4 + W * 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
  cv2.putText(compare_image, '(8192 spp)', (W//4 + W * 2, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

  cv2.putText(compare_image, 'Denoised Image', (W//4 + W, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

  cv2.imshow('Result Image', compare_image)
  if cv2.waitKey() & 0xFF == ord('q'):
    break

  # rp = REFER_IMAGE_DIR + f'{step+1}'
  # dp = DENOISED_IMG_DIR + f'{step+1}'
  # np = NOISY_IMAGE_DIR + f'{step+1}'

  # write_exr(reference[0, :, :, :3], rp + ".exr")
  # write_exr(denoised_img[0, :, :, :3], dp + ".exr")
  # write_exr(noisy_img[0, :, :,:3], np + ".exr")

  # to_jpg(rp + ".exr", rp + ".jpg")
  # to_jpg(dp + ".exr", dp + ".jpg")
  # to_jpg(np + ".exr", np + ".jpg")

  # os.remove(rp + ".exr")
  # os.remove(dp + ".exr")
  # os.remove(np + ".exr")
"""
