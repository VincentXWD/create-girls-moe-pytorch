__author__ = 'Wendong Xu'
import pickle
from scipy import misc
import os
import numpy as np
import utils


aim_size = 128
id = 1
avatar_tag_path = '../../resource/avatar_with_tag.list'
dump_filename = '../../resource/avatar_with_tag.dat'


def get_avatar_with_tag(avatar_tag_path):
  avatar_list = []
  with open(avatar_tag_path, 'r') as fin:
    avatar_list = fin.readlines()
  avatar_list = list(map(lambda each: each.strip('\n'), avatar_list))
  avatar_list = list(map(lambda each: each.split(','), avatar_list))
  avatar_list = list(map(lambda each: [each[0], each[1], each[2].split(';'), each[3]], avatar_list))
  # id, years, tags, path
  return avatar_list


def process_image(img):
  global id
  # resization
  img = misc.imresize(img, [aim_size, aim_size, 3])
  print('{} finished.'.format(id))
  id += 1
  return img


def dump_file(obj, dump_filename):
  with open(dump_filename, 'wb') as fout:
    pickle.dump(obj, fout)


if __name__ == '__main__':
  avatar_list = get_avatar_with_tag(avatar_tag_path)
  result_list = []
  for i, each in enumerate(avatar_list):
    if os.path.exists(each[3]):
      if int(each[1]) < 2005:
        continue
      # tag's one-hot, image-bytes
      result_list.append([utils.get_one_hot(each[2]), process_image(misc.imread(each[3]))])
  dump_file(result_list, dump_filename)
