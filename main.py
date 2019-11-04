import os
import random
import tensorflow as tf
import src.utils as utils
import shutil
import numpy as nps
#from PIL import Image
from itertools import count
from src.model import DenoisingNet
from src.dataset import next_batch_tensor
from src.base_model import SSIM
from src.exr import write_exr, to_jpg
import time
import gc
from tensorflow.python.util import deprecation

# 모델의 이름에 따라 checkpoint나 summary 폴더 저장함.(debug 폴더에서)
EXPERIMENT_NAME   = '4'

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
INPUT_CH          = 44     # model에 입력으로 들어갈 실제 채널 수 (전처리가 되어 늘어남)
OUTPUT_CH         = 3      # 모델의 출력으로 나올 채널 수 (color 3)

TRAIN_TFRECORD    = './data/train.tfrecord'
VALID_TFRECORD    = './data/valid.tfrecord'

LOG_PERIOD        = 100    # 이 주기 마다 summary
SAVE_PERIOD       = 2500   # 이 주기 마다 모델 가중치 저장
VALID_PERIOD      = 1000   # 이 주기 마다 검증(검증 데이터로 inference하고 이미지 저장)

SUMMARY_SCOPE     = 'Color_output'  # summary scope


def main(_):

  # Denoising Model. src\base_model.py과 src\model.py를 참조할 것
  net_Diffuse = DenoisingNet(input_shape=[None, None, INPUT_CH],
                     output_shape=[None, None, OUTPUT_CH],
                     loss_func='L1',
                     start_lr=1e-4,
                     lr_decay_step=10000,
                     lr_decay_rate=1.0,
                     diff_spec_select="Diffuse")

  net_Specular = DenoisingNet(input_shape=[None, None, INPUT_CH],
                     output_shape=[None, None, OUTPUT_CH],
                     loss_func='L1',
                     start_lr=1e-4,
                     lr_decay_step=10000,
                     lr_decay_rate=1.0,
                     diff_spec_select="Specular")

  init = tf.initialize_all_variables()

  with tf.Session() as sess:
    sess.run(init)
    #sess.graph.finalize()
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
    # train_writer = tf.summary.FileWriter(TRAIN_SUMMARY_DIR, sess.graph)
    # valid_writer = tf.summary.FileWriter(VALID_SUMMARY_DIR, sess.graph)
    # =========================================================================

    # 검증용 데이터 tensor 생성 src\dataset 참조
    # =========================================================================
    valid_noisy_img_diffuse, valid_noisy_img_specular, valid_reference_diffuse, valid_reference_specular = \
      next_batch_tensor(
        tfrecord_path = VALID_TFRECORD,
        shape = [IMG_HEIGHT, IMG_WIDTH, N_NOISY_FEATURE, N_REFER_FEATURE],
        repeat = 1000
      )
    # =========================================================================
    
    merged_summary = tf.summary.merge_all()
    
    # 학습 시작 Epoch
    for epoch in count():
      # 학습용 데이터 tensor 생성 src\dataset 참조
      train_noisy_img_diffuse, train_noisy_img_specular, train_reference_diffuse, train_reference_specular = \
        next_batch_tensor(
          tfrecord_path = TRAIN_TFRECORD,
          shape = [PATCH_SIZE, PATCH_SIZE, N_NOISY_FEATURE, N_REFER_FEATURE],
          batch_size = BATCH_SIZE,
          shuffle_buffer = BATCH_SIZE * 2
        )

      lst = []
      # 학습용 데이터 다 읽을 때 까지
      while True:
        # 더이상 읽을 것이 없으면 빠져 나오고 다음 epoch으로
        try:
          # 학습용 데이터를 위에 설정한 베치사이즈만큼 가져온다.
          noisy_img_diffuse, noisy_img_specular, reference_diffuse, reference_specular = \
              sess.run([train_noisy_img_diffuse, train_noisy_img_specular, train_reference_diffuse, train_reference_specular])

        except tf.errors.OutOfRangeError as e:
          print("Done")
          break

        # 데이터로 학습
        _, step, lr = sess.run([net_Diffuse.train_op, net_Diffuse.global_step, net_Diffuse.lr],
                              feed_dict={net_Diffuse.inputs: noisy_img_diffuse,
                                          net_Diffuse.refers: reference_diffuse})

        _, step, lr = sess.run([net_Specular.train_op, net_Specular.global_step, net_Specular.lr],
                              feed_dict={net_Specular.inputs: noisy_img_specular,
                                          net_Specular.refers: reference_specular})
        
        # 일정
        if step % LOG_PERIOD == LOG_PERIOD - 1:
          loss_diffuse = sess.run([net_Diffuse.loss],
                                    feed_dict={net_Diffuse.inputs: noisy_img_diffuse,
                                              net_Diffuse.refers: reference_diffuse})

          loss_specular= sess.run([net_Specular.loss],
                                    feed_dict={net_Specular.inputs: noisy_img_specular,
                                              net_Specular.refers: reference_specular})
          #train_writer.add_summary(summary_diffuse, step + 1)
          #train_writer.add_summary(summary_specular, step + 1)

          print(f"epoch {epoch}, step {step + 1} loss diffuse: {loss_diffuse}, loss specular: {loss_specular}, learning rate : {lr:.7f}")


        if step % SAVE_PERIOD == SAVE_PERIOD - 1:
          saver.save(sess, os.path.join(CKPT_DIR, "model.ckpt"), global_step = step + 1, write_meta_graph=False)

        if step % VALID_PERIOD == VALID_PERIOD - 1:
          noisy_img_diffuse, noisy_img_specular, reference_diffuse, reference_specular = \
              sess.run([valid_noisy_img_diffuse, valid_noisy_img_specular, valid_reference_diffuse, valid_reference_specular])

          
          T = time.time()
          loss_diffuse, denoised_img_diffuse = sess.run([net_Diffuse.loss, net_Diffuse.outputs],
                                        feed_dict={net_Diffuse.inputs:noisy_img_diffuse,
                                                          net_Diffuse.refers:reference_diffuse})

          loss_specular, denoised_img_Specular = sess.run([net_Specular.loss, net_Specular.outputs],
                                        feed_dict={net_Specular.inputs:noisy_img_specular,
                                                  net_Specular.refers:reference_specular})
          infer = time.time() - T 
          lst.append(infer)
          if len(lst) > 1:
            print(f'Inference Time = {nps.mean(lst[1:])}', end= '  ')

          print(" Test ] Diffuse Loss " , loss_diffuse, end= '  ')
          print(" Test ] Specular Loss ", loss_specular, end= '  ')

          denoised_img  = denoised_img_diffuse + denoised_img_Specular
          reference_img = reference_specular   + reference_diffuse

          L2 = nps.square(nps.subtract(denoised_img, reference_img))
          denom = nps.square(reference_img) + 1.0e-2
          loss = nps.mean(L2 / denom)

          print(f" Test ] RelMSE {loss:.4f}")
          collected = gc.collect()
          print(f"Garbage collector: collected {collected} objects.")

          rp = REFER_IMAGE_DIR  + f'{step+1}'
          dp = DENOISED_IMG_DIR + f'{step+1}'
          np = NOISY_IMAGE_DIR  + f'{step+1}'

          write_exr(reference_img[0, :, :, :3],  rp + ".exr")
          write_exr(denoised_img[0,:,:,:3],      dp + ".exr")
          write_exr(noisy_img_diffuse[0,:,:,:3], np + ".exr")

          to_jpg(rp + ".exr", rp + ".jpg")
          to_jpg(dp + ".exr", dp + ".jpg")
          to_jpg(np + ".exr", np + ".jpg")

          os.remove(rp + ".exr")
          os.remove(dp + ".exr")
          os.remove(np + ".exr")

  tf.reset_default_graph()



if __name__ == '__main__':
  deprecation._PRINT_DEPRECATION_WARNINGS = False
  # Winograd 알고리즘(conv 성능 향상)
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # tensorflow logging level 설정
  tf.logging.set_verbosity(tf.logging.INFO)

  tf.app.run()
