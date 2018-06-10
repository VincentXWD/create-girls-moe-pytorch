import numpy as np


feature = ['blonde hair','brown hair','black hair','blue hair','pink hair',
               'purple hair','green hair','red hair','silver hair','white hair','orange hair',
               'aqua hair','gray hair','long hair','short hair','twintails','drill hair','ponytail','blush',
               'smile','open mouth','hat','ribbon','glasses','blue eyes','red eyes','brown eyes',
               'green eyes','purple eyes','yellow eyes','pink eyes','aqua eyes','black eyes','orange eyes',]
feature_map = dict()
for i, j in enumerate(feature):
  feature_map[j] = i


def get_one_hot(feat: list) -> np.array:
  """

  :param feat:
  :return:
  """
  one_hot = np.zeros(len(feat))
  one_hot[feat] = 1
  return one_hot


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

