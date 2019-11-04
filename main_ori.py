import os
import tensorflow as tf
import cv2
import src.utils as utils
import shutil
import numpy as nps
from itertools import count
from src_ori.exr import write_exr, to_jpg
from src_ori.model import DenoisingNet
from src_ori.dataset import next_batch_tensor
import time
# 모델의 이름에 따라 checkpoint나 summary 폴더 저장함.(debug 폴더에서)
EXPERIMENT_NAME   = '1x1_2'

# summary나 checkpoint를 저장할 폴더
DEBUG_DIR         = './debug/'

# 학습 중 이미지가 잘 나오는지 보기 위한 폴더
DEBUG_IMAGE_DIR   = DEBUG_DIR + 'images/' + EXPERIMENT_NAME + '/'
NOISY_IMAGE_DIR   = os.path.join(DEBUG_IMAGE_DIR, "noisy_img/")
REFER_IMAGE_DIR   = os.path.join(DEBUG_IMAGE_DIR, "reference/")
DENOISED_IMG_DIR  = os.path.join(DEBUG_IMAGE_DIR, "denoised_img/")

# 학습용 summary와 검증용 summary 저장하는 곳
TRAIN_SUMMARY_DIR = DEBUG_DIR + 'summary/' + EXPERIMENT_NAME + '/train'
VALID_SUMMARY_DIR = DEBUG_DIR + 'summary/' + EXPERIMENT_NAME + '/valid'

# 모델 저장하기 위한 폴더
CKPT_DIR          = DEBUG_DIR + 'checkpoint/' + EXPERIMENT_NAME + '/'

# 하이퍼 파라미터
BATCH_SIZE        = 32
IMG_HEIGHT        = 720
IMG_WIDTH         = 1280
PATCH_SIZE        = 64
N_NOISY_FEATURE   = 22     # tfrecord에서 데이터 가져올 떄 noisy image의 channel 수(color 3, albedo 3, specular 3, ...)
N_REFER_FEATURE   = 9      # tfrecord에서 데이터 가져올 때 reference의 channel 수(color 3, specular 3, diffuse 3)
INPUT_CH          = 63     # model에 입력으로 들어갈 실제 채널 수 (전처리가 되어 늘어남)
OUTPUT_CH         = 3      # 모델의 출력으로 나올 채널 수 (color 3)

TRAIN_TFRECORD    = './data/train.tfrecord'
VALID_TFRECORD    = './data/valid.tfrecord'

LOG_PERIOD        = 100    # 이 주기 마다 summary
SAVE_PERIOD       = 5000   # 이 주기 마다 모델 가중치 저장
VALID_PERIOD      = 2000   # 이 주기 마다 검증(검증 데이터로 inference하고 이미지 저장)

SUMMARY_SCOPE     = 'Color_output'  # summary scope


def main(_):

  # Denoising Model. src\base_model.py과 src\model.py를 참조할 것
  net = DenoisingNet(input_shape=[None, None, INPUT_CH],
                     output_shape=[None, None, OUTPUT_CH],
                     loss_func='L1',
                     start_lr=1e-6,
                     lr_decay_step=10000,
                     lr_decay_rate=1.0)

  sess = tf.Session()

  # debug용 폴더들 생성
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

  # 검증용 데이터 tensor 생성 src\dataset 참조
  # =========================================================================
  valid_noisy_img, valid_reference = \
    next_batch_tensor(
      tfrecord_path = VALID_TFRECORD,
      shape = [IMG_HEIGHT, IMG_WIDTH, N_NOISY_FEATURE, N_REFER_FEATURE],
      repeat = 1000
    )
  # =========================================================================

  #merged_summary = tf.summary.merge_all()
  lst=[]
  # 학습 시작 Epoch
  for epoch in count():
    # 학습용 데이터 tensor 생성 src\dataset 참조
    train_noisy_img, train_reference = \
      next_batch_tensor(
        tfrecord_path = TRAIN_TFRECORD,
        shape = [PATCH_SIZE, PATCH_SIZE, N_NOISY_FEATURE, N_REFER_FEATURE],
        batch_size = BATCH_SIZE,
        shuffle_buffer = BATCH_SIZE * 2,
      )

    # 학습용 데이터 다 읽을 때 까지
    while True:
      # 더이상 읽을 것이 없으면 빠져 나오고 다음 epoch으로
      try:
        # 학습용 데이터를 위에 설정한 베치사이즈만큼 가져온다.
        noisy_img, reference = sess.run([train_noisy_img, train_reference])
      except tf.errors.OutOfRangeError as e:
        print("Done")
        break

      # 데이터로 학습
      _, step, lr = sess.run([net.train_op, net.global_step, net.lr],
                             feed_dict={net.inputs: noisy_img,
                                        net.refers: reference})
      
      # 일정
      if step % LOG_PERIOD == LOG_PERIOD - 1:
        loss = sess.run([net.loss],
                                  feed_dict={net.inputs: noisy_img,
                                             net.refers: reference})
        # train_writer.add_summary(summary, step + 1)
        print(f'epoch:{epoch:2d} step:{step+1} loss:{loss}')

      if step % SAVE_PERIOD == SAVE_PERIOD - 1:
        saver.save(sess, os.path.join(CKPT_DIR, "model.ckpt"), global_step = step + 1)

      if step % VALID_PERIOD == VALID_PERIOD - 1:
        noisy_img, reference = sess.run([valid_noisy_img, valid_reference])

        loss, denoised_img = sess.run([net.loss, net.outputs],
                                              feed_dict={net.inputs:noisy_img,
                                                         net.refers:reference})

        print(" Test ] Loss ", loss)

        #valid_writer.add_summary(summary, step + 1)
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
