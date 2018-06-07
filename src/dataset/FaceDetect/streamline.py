import argparse
import utils
import logging
from PIL import Image
import os


def remove_below_specified_resolution(raw_img_dir: str, resolution: tuple) -> None:
  """
  remove the image which is smaller than specified resolution.
  :param raw_img_dir:
  :return:
  """
  img_paths = utils.get_image_path(raw_img_dir)
  for path in img_paths:
    print(path)
    size = Image.open(path).size
    if size[0] < resolution[0] or size[0] < resolution[1]:
      os.remove(path)
      print('{} doesn\'t satisfied. deleted.'.format(path))


if __name__ == '__main__':
  logging.basicConfig(filename=None, level=logging.INFO, format='%(levelname)s:%(message)s',
                      datefmt='%d-%m-%Y %I:%M:%S %p')
  logging.getLogger().addHandler(logging.StreamHandler())

  parser = argparse.ArgumentParser()
  parser.add_argument("--raw_img_dir", type=str,
                      default="../../../resource/getchu_avatar/",
                      help='''''')
  parser.add_argument("--resolution", type=tuple,
                      default=(42, 42),
                      help='''''')
  FLAGS, unparsed = parser.parse_known_args()
  raw_img_dir = FLAGS.raw_img_dir
  resolution = FLAGS.resolution

  remove_below_specified_resolution(raw_img_dir, resolution)
