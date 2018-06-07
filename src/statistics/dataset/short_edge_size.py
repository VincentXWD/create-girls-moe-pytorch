import utils
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image


def get_short_size(image_path: str) -> int:
  size = Image.open(image_path).size
  return min(size[0], size[1])


def get_short_edge_size(avatar_list_path: str, getchu_data_path: str) -> list:
  """
  statistics of dataset's shortest edge's size.
  :param avatar_list_path:
  :param getchu_data_path:
  :return:
  """
  avatar_list = utils.read_list(avatar_list_path)
  getchu_data_list = utils.read_list(getchu_data_path)
  avatar_list = list(map(lambda each: get_short_size(each[2].strip('\n')), avatar_list))
  statistics = [0 for i in range(0, np.max(np.array(list(map(lambda each: each, avatar_list))))+1)]
  for each in avatar_list:
    statistics[each] += 1
  print(statistics[42:])


if __name__ == '__main__':
  statistics = get_short_edge_size('../../../resource/avatar.list', '../../../resource/getchu_datas.txt')
