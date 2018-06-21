__author__ = 'Wendong Xu'
import numpy as np


hair = ['blonde hair','brown hair','black hair','blue hair','pink hair',
        'purple hair','green hair','red hair','silver hair','white hair','orange hair',
        'aqua hair','gray hair']
eyes = ['blue eyes','red eyes','brown eyes',
        'green eyes','purple eyes','yellow eyes','pink eyes','aqua eyes','black eyes',
        'orange eyes']
others = ['long hair','short hair','twintails','drill hair','ponytail','blush',
          'smile','open mouth','hat','ribbon','glasses']
tag = ['blonde hair','brown hair','black hair','blue hair','pink hair',
               'purple hair','green hair','red hair','silver hair','white hair','orange hair',
               'aqua hair','gray hair','long hair','short hair','twintails','drill hair','ponytail','blush',
               'smile','open mouth','hat','ribbon','glasses','blue eyes','red eyes','brown eyes',
               'green eyes','purple eyes','yellow eyes','pink eyes','aqua eyes','black eyes','orange eyes',]
tag_map = dict()
for i, j in enumerate(tag):
  tag_map[j] = i


def get_one_hot(feat: list) -> np.array:
  """

  :param feat:
  :return:
  """
  one_hot = np.zeros(len(tag))
  one_hot[list(map(lambda each: tag_map[each], feat))] = 1
  return one_hot


if __name__ == '__main__':
  print(get_one_hot(['green eyes', 'smile']))