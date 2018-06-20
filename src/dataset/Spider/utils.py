__author__ = 'Wendong Xu'
import os


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