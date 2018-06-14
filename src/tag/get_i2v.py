import i2v
from PIL import Image
import argparse
import logging
import os
import config
import numpy as np
import utils
from functools import reduce
import shutil


# __DEBUG__ = True
__DEBUG__ = False


def get_main_tag(img: Image) -> np.array:
  """

  :param img: PIL.Image
  :return: feature list
  """
  illust2vec_tag = i2v.make_i2v_with_chainer(config.tag_model_path, config.tag_list_json)

  result = illust2vec_tag.estimate_plausible_tags([img], threshold=config.tag_threshold)[-1]['general']

  feat = []
  for each in result:
    if each[0] in utils.feature_map.keys():
      feat.append(each[0])
  return feat


def batchmark(avatar_list_path: str, getchu_data_path: str, list_output_path: str, avatar_output_path: str) -> None:
  """

  :return:
  """
  years = utils.get_release_years_hot_map(avatar_list_path, getchu_data_path)
  with open(avatar_list_path) as fin:
    avatar_path = fin.readlines()
  avatar_path = list(map(lambda each: each.split(' '), avatar_path))
  avatar_path = list(map(lambda each: [each[0], each[-1].strip('\n')], avatar_path))
  fz = len(str(len(avatar_path)))
  new_id = 0
  with open(list_output_path, 'w') as fout:
    for each in avatar_path:
      file_path = each[-1][3:]
      id = each[0]
      img = Image.open(file_path)
      feat = get_main_tag(img)
      if len(feat) == 0:
        continue
      feat = reduce(lambda x, y: x + ';' + y, feat)
      save_path = os.path.join(avatar_output_path, str(new_id).zfill(fz)+'.jpg')
      print('{},{},{},{}'.format(str(new_id).zfill(fz), years[int(id)], feat, save_path))
      fout.write('{},{},{},{}\n'.format(str(new_id).zfill(fz), years[int(id)], feat, save_path))
      shutil.copyfile(file_path, save_path)
      new_id += 1


def main():
  logging.basicConfig(filename=None, level=logging.INFO, format='%(levelname)s:%(message)s',
                      datefmt='%d-%m-%Y %I:%M:%S %p')
  logging.getLogger().addHandler(logging.StreamHandler())

  parser = argparse.ArgumentParser()
  parser.add_argument("--avatar_list_path", type=str,
                      default="../../resource/avatar.list",
                      help='''''')
  parser.add_argument("--getchu_data_path", type=str,
                      default="../../resource/getchu_datas.txt",
                      help='''''')
  parser.add_argument("--list_output_path", type=str,
                      default="../../resource/avatar_with_tag.list",
                      help='''''')
  parser.add_argument("--avatar_output_path", type=str,
                      default="../../resource/avatar_with_tag/",
                      help='''''')
  FLAGS, unparsed = parser.parse_known_args()
  logging.info('--avatar_list_path: {}'.format(os.path.abspath(FLAGS.avatar_list_path)))
  logging.info('--list_output_path : {}'.format(os.path.abspath(FLAGS.list_output_path)))
  logging.info('--getchu_data_path : {}'.format(os.path.abspath(FLAGS.getchu_data_path)))
  logging.info('--avatar_output_path : {}'.format(os.path.abspath(FLAGS.avatar_output_path)))

  if __DEBUG__:
    img = Image.open("../../resource/avatar_with_tag/00001.jpg")
    print(get_main_tag(img))
  else:
    if not os.path.exists(FLAGS.avatar_output_path):
      os.mkdir(FLAGS.avatar_output_path)
    batchmark(FLAGS.avatar_list_path, FLAGS.getchu_data_path, FLAGS.list_output_path, FLAGS.avatar_output_path)


if __name__ == '__main__':
  main()
