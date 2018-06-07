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


def read_avatar_list(avatar_list_path: str) -> list:
  """
  [id, idx, avatar_path]
  :param avatar_list_path:
  :return:
  """
  avatar_list = []
  with open(avatar_list_path) as fin:
    avatar_list = fin.readlines()
  avatar_list = list(map(lambda x: x.split(' '), avatar_list))
  return avatar_list


def update_avatar_list(avatar_dir: str, avatar_list_path: str) -> None:
  """
  update the avatar list to make sync with getchu_avatar
  :param avatar_dir:
  :param avatar_list_path:
  :return:
  """
  avatar_list = read_avatar_list(avatar_list_path)
  avatar_image_path = get_image_path(avatar_dir)
  ptn = re.compile('(\d+)_(\d+).jpg')
  with open(avatar_list_path, 'w') as fout:
    for image_path in avatar_image_path:
      i, id_x = re.findall(ptn, image_path)[-1]
      fout.write('{} {} {}\n'.format(i, id_x, image_path))


if __name__ == '__main__':
  # to update the avatar list
  update_avatar_list('../../../resource/getchu_avatar/', '../../../resource/avatar.list')
