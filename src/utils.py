import os
import shutil

def make_dir(dir_name):
  if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

def remove_dir(dir_name):
  if os.path.isdir(dir_name):
    shutil.rmtree(dir_name)
