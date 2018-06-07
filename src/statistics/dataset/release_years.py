import utils
import numpy as np
import re
import matplotlib.pyplot as plt


def get_release_years(avatar_list_path: str, getchu_data_path: str) -> list:
  """
  statistics of dataset's release years.
  :param avatar_list_path:
  :param getchu_data_path:
  :return:
  """
  avatar_list = utils.read_list(avatar_list_path)
  getchu_data_list = utils.read_list(getchu_data_path)
  avatar_list = list(map(lambda each: int(each[0]), avatar_list))

  getchu_data_list = list(map(lambda each: (int(each[0]), int(re.findall('(\d+)-\d+-\d+', each[1])[-1])), getchu_data_list))
  years = [0 for i in range(0, np.max(np.array(list(map(lambda each: each[0], getchu_data_list))))+1)]
  statistics = [0 for i in range(0, np.max(np.array(list(map(lambda each: each[1], getchu_data_list))))+1)]
  for each in getchu_data_list:
    years[each[0]] = each[1]
  for each in avatar_list:
    statistics[years[each]] += 1
  print(statistics[1990:])
  return statistics


if __name__ == '__main__':
  statistics = get_release_years('../../../resource/avatar.list', '../../../resource/getchu_datas.txt')[1990:]
