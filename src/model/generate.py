import argparse
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import os
from networks.generator import Generator
import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="")
parser.add_argument('--avatar_tag_dat_path', type=str, default='../../resource/avatar_with_tag.dat', help='avatar with tag\'s list path')
parser.add_argument('--tmp_path', type=str, default='../../resource/training_temp/', help='path of the intermediate files during training')
parser.add_argument('--model_dump_path', type=str, default='../../resource/gan_models', help='model\'s save path')

opt = parser.parse_args()
tmp_path= opt.tmp_path
model_dump_path = opt.model_dump_path


def load_checkpoint(model_dir):
  models_path = utils.read_newest_model(model_dir)
  if len(models_path) == 0:
    return None, None
  models_path.sort()
  new_model_path = os.path.join(model_dump_path, models_path[-1])
  checkpoint = torch.load(new_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
  return checkpoint, new_model_path


def generate_test(G, file_name):
  g_noise, fake_tag = utils.fake_generator(1, 128, device)
  fake_feat = torch.cat([g_noise, fake_tag], dim=1)
  fake_img = G(fake_feat)
  vutils.save_image(fake_img.data.view(1, 3, 128, 128),
                    os.path.join(tmp_path, '{}.png'.format(file_name)))
  print('Saved intermediate file in {}'.format(os.path.join(tmp_path, '{}.png'.format(file_name))))


if __name__ == '__main__':
  G = Generator().to(device)
  checkpoint, _ = load_checkpoint(model_dump_path)
  G.load_state_dict(checkpoint['G'])
  generate_test(G, 'test')
