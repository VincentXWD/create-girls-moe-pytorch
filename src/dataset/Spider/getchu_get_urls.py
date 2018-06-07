import argparse
import re
import logging
import os
import pyquery


ID_PATTERN = re.compile('id=(\d+)')
DATA_PATTERN = re.compile('\d*-\d*-\d+')


def get_url_and_date(I: str, O: str, id_data_output_path: str) -> None:
  '''
  Get image url and date.
  Saved in the resource directory with names of `O` and `id_data_output_path`.
  :param I:
  :param O:
  :param id_data_output_path:
  :return: None
  '''
  with open(I, encoding='utf-8') as fin:
    doc = pyquery.PyQuery(fin.read())
  table = doc.attr('id', 'query_result_main')('tbody')
  id_data = []
  with open(O, 'w', encoding='utf-8') as fout:
    for line in table.items():
      for tr in line('tr').items():
        lst = re.findall(ID_PATTERN, tr.text())
        data = re.findall(DATA_PATTERN, tr.text())
        if len(lst) == 0:
          continue
        fout.write('http://www.getchu.com/soft.phtml?id={}&gc=gc\n'.format(lst[-1]))
        id_data.append([lst[-1], data[-1]])
  with open(id_data_output_path, 'w', encoding='utf-8') as fout:
    for each in id_data:
      fout.write('{} {}\n'.format(each[0], each[1]))


def main():
  logging.basicConfig(filename=None, level=logging.INFO, format='%(levelname)s:%(message)s',
                      datefmt='%d-%m-%Y %I:%M:%S %p')
  logging.getLogger().addHandler(logging.StreamHandler())

  parser = argparse.ArgumentParser()
  parser.add_argument("--input_path", type=str,
                      default="../../../resource/erogamescape_sql_output/results.html",
                      help='''''')
  parser.add_argument("--urls_output_path", type=str,
                      default="../../../resource/getchu_urls.txt",
                      help='''''')
  parser.add_argument("--id_data_output_path", type=str,
                      default="../../../resource/getchu_datas.txt",
                      help='''''')
  FLAGS, unparsed = parser.parse_known_args()
  logging.info('--input_path: {}'.format(os.path.abspath(FLAGS.input_path)))
  logging.info('--urls_output_path : {}'.format(os.path.abspath(FLAGS.urls_output_path)))

  get_url_and_date(FLAGS.input_path, FLAGS.urls_output_path, FLAGS.id_data_output_path)


if __name__ == '__main__':
  main()