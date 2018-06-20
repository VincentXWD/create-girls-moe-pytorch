__author__ = 'Wendong Xu'
import numpy as np
import re


tag = ['blonde hair','brown hair','black hair','blue hair','pink hair',
               'purple hair','green hair','red hair','silver hair','white hair','orange hair',
               'aqua hair','gray hair','long hair','short hair','twintails','drill hair','ponytail','blush',
               'smile','open mouth','hat','ribbon','glasses','blue eyes','red eyes','brown eyes',
               'green eyes','purple eyes','yellow eyes','pink eyes','aqua eyes','black eyes','orange eyes',]
tag_map = dict()
for i, j in enumerate(tag):
  tag_map[j] = i


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


def get_release_years_hot_map(avatar_list_path: str, getchu_data_path: str) -> list:
  """
  get release years' hotmap.
  :param avatar_list_path:
  :param getchu_data_path:
  :return:
  """
  avatar_list = read_list(avatar_list_path)
  getchu_data_list = read_list(getchu_data_path)
  avatar_list = list(map(lambda each: int(each[0]), avatar_list))

  getchu_data_list = list(map(lambda each: (int(each[0]), int(re.findall('(\d+)-\d+-\d+', each[1])[-1])), getchu_data_list))
  years = [0 for i in range(0, np.max(np.array(list(map(lambda each: each[0], getchu_data_list))))+1)]

  for each in getchu_data_list:
    years[each[0]] = each[1]
  return years


if __name__ == '__main__':
  print(np.count_nonzero(get_release_years_hot_map('../../resource/avatar.list', '../../resource/getchu_datas.txt')))