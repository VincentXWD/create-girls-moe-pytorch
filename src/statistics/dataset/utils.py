import os
import re


def get_image_path(raw_input_dir: str) -> list:
  """
  get image path and id from root resource path.
  :return: a list contains all images' path.
  """
  result = []
  for root, dirs, files in os.walk(raw_input_dir):
    for file in files:
      result.append(os.path.join(root, file))
  return result


def read_list(list_path: str) -> list:
  """

  :param list_path:
  :return:
  """
  avatar_list = []
  with open(list_path) as fin:
    avatar_list = fin.readlines()
  avatar_list = list(map(lambda x: x.split(' '), avatar_list))
  return avatar_list

