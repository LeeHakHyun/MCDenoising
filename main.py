import os
import tensorflow as tf
import cv2
import src.utils as utils
import shutil
from itertools import count
from src.exr import write_exr, to_jpg
from src.model import DenoisingNet
from src.dataset import next_batch_tensor

# TODO: 두 가지 실험하자.
# 1. 모델 하나로 Color direct 출력. 전처리 없이
# 2. 모델 두개로 Diffuse, specular 두개로 분리해서 여러가지 전처리해서 처리

EXPERIMENT_NAME   = 'Model10'

BATCH_SIZE        = 32
IMG_HEIGHT        = 720
IMG_WIDTH         = 1280
PATCH_SIZE        = 64
N_NOISY_FEATURE   = 22
N_REFER_FEATURE   = 9
INPUT_CH          = 57
OUTPUT_CH         = 3

TRAIN_TFRECORD    = './data/train.tfrecord'
VALID_TFRECORD    = './data/valid.tfrecord'

DEBUG_DIR         = './debug/'

DEBUG_IMAGE_DIR   = DEBUG_DIR + 'images/' + EXPERIMENT_NAME + '/'
NOISY_IMAGE_DIR   = os.path.join(DEBUG_IMAGE_DIR, "noisy_img/")
REFER_IMAGE_DIR   = os.path.join(DEBUG_IMAGE_DIR, "reference/")
DENOISED_IMG_DIR  = os.path.join(DEBUG_IMAGE_DIR, "denoised_img/")

TRAIN_SUMMARY_DIR = DEBUG_DIR + 'summary/' + EXPERIMENT_NAME + '/train'
VALID_SUMMARY_DIR = DEBUG_DIR + 'summary/' + EXPERIMENT_NAME + '/valid'
CKPT_DIR          = DEBUG_DIR + 'checkpoint/' + EXPERIMENT_NAME + '/'

LOG_PERIOD        = 100
SAVE_PERIOD       = 5000
VALID_PERIOD      = 2000

SUMMARY_SCOPE     = 'Color_output'


def main(_):

  net = DenoisingNet(input_shape=[None, None, INPUT_CH],
                     output_shape=[None, None, OUTPUT_CH],
                     loss_func='L1',
                     start_lr=0.0001,
                     lr_decay_step=10000,
                     lr_decay_rate=1.0)

  sess = tf.Session()

  utils.make_dir(TRAIN_SUMMARY_DIR)
  utils.make_dir(VALID_SUMMARY_DIR)
  utils.make_dir(CKPT_DIR)
  utils.make_dir(NOISY_IMAGE_DIR)
  utils.make_dir(REFER_IMAGE_DIR)
  utils.make_dir(DENOISED_IMG_DIR)

  # Saver
  # =========================================================================
  saver = tf.train.Saver()
  
  print('Saver initialized')
  recent_ckpt_job_path = tf.train.latest_checkpoint(CKPT_DIR)
        
  if recent_ckpt_job_path is None:
    sess.run(tf.global_variables_initializer())
    print("Initializing variables...")
  else:
    saver.restore(sess, recent_ckpt_job_path)
    print("Restoring...", recent_ckpt_job_path)    
  # =========================================================================

  # Summary
  # =========================================================================
  train_writer = tf.summary.FileWriter(TRAIN_SUMMARY_DIR, sess.graph) 
  valid_writer = tf.summary.FileWriter(VALID_SUMMARY_DIR, sess.graph)
  # =========================================================================

  valid_noisy_img, valid_reference = \
    next_batch_tensor(
      tfrecord_path = VALID_TFRECORD,
      shape = [IMG_HEIGHT, IMG_WIDTH, N_NOISY_FEATURE, N_REFER_FEATURE],
      repeat = 1000
    )
  
  
  merged_summary = tf.summary.merge_all()
  for epoch in count():
    train_noisy_img, train_reference = \
      next_batch_tensor(
        tfrecord_path = TRAIN_TFRECORD,
        shape = [PATCH_SIZE, PATCH_SIZE, N_NOISY_FEATURE, N_REFER_FEATURE],
        batch_size = BATCH_SIZE,
        shuffle_buffer = BATCH_SIZE * 2,
      )

    while True:
      # 더이상 읽을 것이 없으면 빠져 나오고 다음 epoch으로
      try:
        noisy_img, reference = sess.run([train_noisy_img, train_reference])
      except tf.errors.OutOfRangeError as e:
        print("Done")
        break      

      _, step, lr = sess.run([net.train_op, net.global_step, net.lr],
                             feed_dict={net.inputs: noisy_img,
                                        net.refers: reference})

      
      if step % LOG_PERIOD == LOG_PERIOD - 1:
        loss, summary = sess.run([net.loss, merged_summary], 
                                  feed_dict={net.inputs: noisy_img,
                                             net.refers: reference})
        train_writer.add_summary(summary, step + 1)

        print(f"epoch {epoch}, step {step + 1}] loss : {loss:.4f}, learning rate : {lr:.7f}")

        
      if step % SAVE_PERIOD == SAVE_PERIOD - 1:
        saver.save(sess, os.path.join(CKPT_DIR, "model.ckpt"), global_step = step + 1)

      if step % VALID_PERIOD == VALID_PERIOD - 1:
        noisy_img, reference = sess.run([valid_noisy_img, valid_reference])

        loss, denoised_img, summary = sess.run([net.loss, net.outputs, merged_summary],
                                              feed_dict={net.inputs:noisy_img,
                                                         net.refers:reference})

        print(" Test ] Loss ", loss)

        valid_writer.add_summary(summary, step + 1)
        rp = REFER_IMAGE_DIR + f'{step+1}'
        dp = DENOISED_IMG_DIR + f'{step+1}'
        np = NOISY_IMAGE_DIR + f'{step+1}'

        write_exr(reference[0, :, :, :3], rp + ".exr")
        write_exr(denoised_img[0,:,:,:3], dp + ".exr")
        write_exr(noisy_img[0,:,:,:3], np + ".exr")

        to_jpg(rp + ".exr", rp + ".jpg")
        to_jpg(dp + ".exr", dp + ".jpg")
        to_jpg(np + ".exr", np + ".jpg")

        os.remove(rp + ".exr")
        os.remove(dp + ".exr")
        os.remove(np + ".exr")
        

  sess.close()
  tf.reset_default_graph()
  


if __name__ == '__main__':
    
  # Winograd 알고리즘(conv 성능 향상)
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # tensorflow logging level 설정
  tf.logging.set_verbosity(tf.logging.INFO)

  tf.app.run()
