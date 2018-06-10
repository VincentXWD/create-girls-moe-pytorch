import i2v
from PIL import Image
import argparse
import logging
import os
import config
import numpy as np


__DEBUG__ = True


def get_tag_and_vec(avatar_msg: list) -> list:
  """

  :param avatar_msg: [id, id_x, avatar_path]
  :return: [id, id_x, avatar_path, tag]
  """
  img = Image.open(avatar_msg[2])
  illust2vec_tag = i2v.make_i2v_with_chainer(config.tag_model_path, config.tag_list_json)

  result = illust2vec_tag.estimate_plausible_tags([img], threshold=config.tag_threshold)
  return avatar_msg.append(result[-1]['general'])


def batchmark():
  """

  :return:
  """
  pass


def main():
  logging.basicConfig(filename=None, level=logging.INFO, format='%(levelname)s:%(message)s',
                      datefmt='%d-%m-%Y %I:%M:%S %p')
  logging.getLogger().addHandler(logging.StreamHandler())

  parser = argparse.ArgumentParser()
  parser.add_argument("--avatar_list_path", type=str,
                      default="../../resource/avatar.list",
                      help='''''')
  parser.add_argument("--list_output_path", type=str,
                      default="../../resource/avatar_with_tag.list",
                      help='''''')
  FLAGS, unparsed = parser.parse_known_args()
  logging.info('--avatar_list_path: {}'.format(os.path.abspath(FLAGS.avatar_list_path)))
  logging.info('--list_output_path : {}'.format(os.path.abspath(FLAGS.list_output_path)))

  if __DEBUG__:
    img = Image.open("../../resource/20180608225656.png")
    illust2vec_tag = i2v.make_i2v_with_chainer(config.tag_model_path, config.tag_list_json)

    result = illust2vec_tag.estimate_plausible_tags([img], threshold=config.tag_threshold)
    print(result[-1]['general'])


if __name__ == '__main__':
  main()
