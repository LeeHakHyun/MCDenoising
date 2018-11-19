import os 
import cv2
import Imath
import OpenEXR
import numpy as np
from scipy.misc import imsave
from PIL import Image
from glob import glob

import src.utils as utils

import tensorflow as tf

channel_list = [  
               'B',          'G',          'R',    'colorVariance.Z', 
      'specular.B', 'specular.G', 'specular.R', 'specularVariance.Z',
       'diffuse.B',  'diffuse.G',  'diffuse.R',  'diffuseVariance.Z', 
        'normal.B',   'normal.G',   'normal.R',   'normalVariance.Z', 
        'albedo.B',   'albedo.G',   'albedo.R',   'albedoVariance.Z', 
         'depth.Z',                                'depthVariance.Z',
  ]

def write_tfrecord(writer, noisy_img, reference):
  # Convert bytes(image) to features (for tfrecord)
  def _bytes_to_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  example = tf.train.Example(
    features=tf.train.Features(feature={
      'image/noisy_img'     : _bytes_to_feature(noisy_img.tostring()),
      'image/reference'     : _bytes_to_feature(reference.tostring()),
      }
    )
  )
  writer.write(example.SerializeToString())

def read_exr(file_name):

  input_file = OpenEXR.InputFile(file_name)

  header     = input_file.header()
  channels   = header['channels']
  dw         = header['dataWindow']

  width      = dw.max.x - dw.min.x + 1
  height     = dw.max.y - dw.min.y + 1

  # channels 값 읽기
  strings   = input_file.channels(channel_list)

  # 값을 저장할 곳
  exr_data  = np.zeros((height, width, len(channel_list)))
  

  # 각 채널의 string을 처리
  for index, string in enumerate(strings):
    # 읽은 string 값을 각 채널의 data type으로 변환(FLOAT, HALF)
    data = np.fromstring(
      string, 
      dtype = np.float32 if str(channels[channel_list[index]].type) == 'FLOAT'
         else np.float16
      )

    # height, width로 변환
    data = data.reshape(height, width)

    # nan-> 0, inf -> big number
    data = np.nan_to_num(data)

    # np.float32로 변환 후 저장
    exr_data[:, :, index] = data

  return exr_data.astype(np.float32)

def write_jpg(data, path):
  data = np.where(data <= 0.0031308,
                  (data * 12.92) * 255.0,
                  (1.055*(data ** (1.0/2.4)) - 0.055) * 255.0)

  imsave(path, data[:, :, ::-1])


def write_exr(data, path):
  ''' data를 exr파일로 저장
  Args:
    path: 경로 + 파일 이름.exr
    data: 3차원 이미지 데이터
  '''

  assert data.ndim == 3, "exr 파일 write하려는데 차원 수가 다름" + str(data.shape)

  utils.make_dir(os.path.dirname(path))

  h, w, c = data.shape

  if c == 3:
    channel_list = ['B', 'G', 'R']
  else:
    channel_list = ['B']

  
  header = OpenEXR.Header(w, h)
  
  header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)

  # header['channels'] = {c: pixel_type#Imath.Channel(Imath.PixelType.FLOAT) 
  #                       for c in channel_list}

  out = OpenEXR.OutputFile(path, header)

  ch_data = {ch: data[:, :, index].astype(np.float32).tostring()
                 for index, ch in enumerate(channel_list)}

  out.writePixels(ch_data)

def to_jpg(exrfile, jpgfile):
  ''' exrfile을 jpgfile로 변환합니다. '''
  
  color_ch = ['R', 'G', 'B']
  
  File    = OpenEXR.InputFile(exrfile)
  PixType = Imath.PixelType(Imath.PixelType.FLOAT)
  DW      = File.header()['dataWindow']
  Size    = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
  rgb     = [np.fromstring(File.channel(c, PixType), dtype=np.float32) \
                           for c in color_ch]
  
  for i in range(len(color_ch)):
      rgb[i] = np.where(rgb[i]<=0.0031308,
              (rgb[i]*12.92)*255.0,
              (1.055*(rgb[i]**(1.0/2.4))-0.055) * 255.0)
  
  rgb8 = [Image.frombytes("F", Size, c.tostring()).convert("L") for c in rgb]
  Image.merge("RGB", rgb8).save(jpgfile, "JPEG", quality=95)

def read_exr_pair(exr_dir):
  noisy_img_files = sorted(glob(os.path.join(exr_dir, "noisy_img", "*.exr")))
  reference_files = sorted(glob(os.path.join(exr_dir, "reference", "*.exr")))

  for noisy_exr, reference_exr in zip(noisy_img_files, reference_files):

    noisy_exr_name = os.path.basename(noisy_exr).split('-')[0]
    reference_name = os.path.basename(reference_exr).split('-')[0]
    
    assert noisy_exr_name == reference_name

    print("Reading..." + noisy_exr_name)

    noisy_img = read_exr(noisy_exr)
    reference = read_exr(reference_exr)    

    # reference는 color와 specular, diffuse만
    color = reference[:, :, :3]
    specular = reference[:, :, 4:7]
    diffuse = reference[:, :, 8:11]

    reference = np.concatenate((color, specular, diffuse), axis=2)

    yield noisy_img, reference

def make_patches(noisy_img, reference, patch_size, n_patch):
  h, w, _ = np.shape(noisy_img)
  max_x = w - (patch_size - 1)
  max_y = h - (patch_size - 1)

  for _ in range(n_patch):
    x, y = np.random.randint(max_x), np.random.randint(max_y)

    # 임의의 x, y좌표를 가지고 crop
    patched_noisy_img = noisy_img[y:y+patch_size, x:x+patch_size, :]
    patched_reference = reference[y:y+patch_size, x:x+patch_size, :]

    yield patched_noisy_img, patched_reference

def generate_patched_data(exr_dir, tfrecord_path, patch_size, n_patch):
  with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
    for noisy_img, reference in read_exr_pair(exr_dir):
      for patch_ni, patch_re in make_patches(noisy_img, reference, patch_size, n_patch):
        write_tfrecord(writer, patch_ni, patch_re)

def generate_data(exr_dir, tfrecord_path):
  with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
    for noisy_img, reference in read_exr_pair(exr_dir):

      write_tfrecord(writer, noisy_img, reference) 

def generate_jpg_data(exr_dir):
  noisy_img_files = sorted(glob(os.path.join(exr_dir, "noisy_img", "*.exr")))
  reference_files = sorted(glob(os.path.join(exr_dir, "reference", "*.exr")))

  for noisy_exr, reference_exr in zip(noisy_img_files, reference_files):

    noisy_exr_name = os.path.basename(noisy_exr).split('-')[0]
    reference_name = os.path.basename(reference_exr).split('-')[0]
    
    to_jpg(noisy_exr, os.path.join(os.path.dirname(noisy_exr), noisy_exr_name + ".jpg"))
    to_jpg(reference_exr, os.path.join(os.path.dirname(reference_exr), reference_name + ".jpg"))

if __name__ == '__main__':
  generate_patched_data('./data/train', './data/train.tfrecord', 64, 500)
  generate_data('./data/valid', './data/valid.tfrecord')
  #generate_jpg_data('./data/valid')