__author__ = 'Wendong Xu'
import numpy as np
import os
import random
from torch.autograd import Variable
import torch


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


def read_newest_model(model_dump_path):
  for rt, dirs, files in os.walk(model_dump_path):
    return files


def fake_tag():
  global tag
  _tag = [0 for _ in range(len(tag))]
  hair_id = random.randint(0, len(hair)-1)
  eye_id = random.randint(0 ,len(eyes)-1)
  _tag[tag_map[hair[hair_id]]] = 1
  _tag[tag_map[eyes[eye_id]]] = 1
  for feat in others:
    choice = random.randint(1, 4)
    if choice == 4:
      _tag[tag_map[feat]] = 1
  _tag = Variable(torch.FloatTensor(_tag)).view(1,-1)
  return _tag


def fake_generator(batch_size, noise_size, device):
  noise = Variable(torch.FloatTensor(batch_size, noise_size)).to(device)
  noise.data.normal_(.0, 1)
  tag = torch.cat([fake_tag() for i in range(batch_size)], dim=0)
  tag = Variable(tag).to(device)
  return noise, tag


if __name__ == '__main__':
  print(get_one_hot(['green eyes', 'smile']))