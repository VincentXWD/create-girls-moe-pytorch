__author__ = 'Wendong Xu'
import cv2
import os
import math
import argparse
import logging
import re
import utils

__DEBUG__ = False
DETECTOR_MODEL_PATH = '../../../resource/animeface_model/lbpcascade_animeface.xml'
ID_PATTERN = re.compile('(\d+)_\d+.jpg')
IDX_PATTERN = re.compile('\d+_(\d+).jpg')


def face_detect(read_path: str, model_path: str=DETECTOR_MODEL_PATH, scale: int=1.5):
  """
  detect anime face in image.
  with resolution limit.
  :param read_path:
  :param model_path:
  :param scale:
  :return:
  """
  if not os.path.isfile(model_path):
    raise RuntimeError("%s: not found" % model_path)
  cascade = cv2.CascadeClassifier(model_path)
  image = cv2.imread(read_path, cv2.IMREAD_COLOR)
  image_copy = image.copy()
  size = image.shape
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.equalizeHist(gray)
  faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
  detect_result = []
  for (x, y, w, h) in faces:
    tx, ty, tw, th = x, y, w, h
    ww, hh = w * scale, h * scale
    W, H = math.ceil(ww), math.ceil(hh)
    w, h = math.ceil((ww - w) / 2), math.ceil((hh - h) / 2)
    x, y = math.ceil(max(0, x - w)), math.ceil(max(0, y - h))
    if x + W >= size[0] or y + H >= size[1]:
      continue
    X, Y = x + W + 1, y + H + 1
    detect_result.append((x, y, X, Y, image_copy[y:Y, x:X, :]))
    if __DEBUG__:
      cv2.rectangle(image, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 2)
      cv2.rectangle(image, (x, y), (X, Y), (0, 0, 255), 2)
  if __DEBUG__:
    cv2.imshow('D', image)
    cv2.waitKey()
  return detect_result


def batchmark(img_paths: list, output_dir: str, output_list_path: str) -> None:
  """
  batchmark of face detecting.
  :param img_paths:
  :return: a list which has this form: [id, id_x, [ROI], path]
  """
  with open(output_list_path, 'w') as fout:
    for path in img_paths:
      id = re.findall(ID_PATTERN, path)[-1]
      id_x = re.findall(IDX_PATTERN, path)[-1]
      ROI = face_detect(path)
      if len(ROI) == 0:
        continue
      for roi in ROI:
        avatar_path = os.path.join(output_dir, id + '_' + id_x + '.jpg')
        fout.write('{} {} {}\n'.format(id, id_x, avatar_path))
        print('{} {} {}'.format(id, id_x, avatar_path))
        cv2.imwrite(avatar_path, roi[-1])


if __name__ == '__main__':
  logging.basicConfig(filename=None, level=logging.INFO, format='%(levelname)s:%(message)s',
                      datefmt='%d-%m-%Y %I:%M:%S %p')
  logging.getLogger().addHandler(logging.StreamHandler())

  parser = argparse.ArgumentParser()
  parser.add_argument("--raw_input_dir", type=str,
                      default="../../../resource/getchu_raw_img/",
                      help='''''')

  parser.add_argument("--output_dir", type=str,
                      default="../../../resource/getchu_avatar/",
                      help='''''')

  parser.add_argument("--output_list_path", type=str,
                      default="../../../resource/avatar.list",
                      help='''''')
  FLAGS, unparsed = parser.parse_known_args()
  raw_input_dir = FLAGS.raw_input_dir
  output_dir = FLAGS.output_dir
  output_list_path = FLAGS.output_list_path

  logging.info('--raw_input_dir: {}'.format(os.path.abspath(FLAGS.raw_input_dir)))
  logging.info('--output_dir : {}'.format(os.path.abspath(FLAGS.output_dir)))
  logging.info('--output_list_path : {}'.format(os.path.abspath(FLAGS.output_list_path)))

  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  if __DEBUG__:
    face_detect('../../../resource/getchu_raw_img/880916_13.jpg')

  # img_paths = utils.get_image_path(raw_input_dir)
  # batchmark(img_paths, output_dir, output_list_path)