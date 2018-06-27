__author__ = 'Wendong Xu'
'''
Training strategy of the `DRAGAN-like SRGAN`.
Have some difference in loss calculating.
I weighted label's loss and tag's loss with half of lambda_adv.
The label_criterion was also different.
'''
import argparse
from networks.generator import Generator
from networks.discriminator import Discriminator
from data_loader import AnimeFaceDataset
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
import utils
import random
import os
import torchvision.utils as vutils
import logging
import time


__DEBUG__ = True

# have GPU or not.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet-GAN")
parser.add_argument('--avatar_tag_dat_path', type=str, default='../../resource/avatar_with_tag.dat', help='avatar with tag\'s list path')

parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.5, help='adam optimizer\'s paramenter')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size for each epoch')
parser.add_argument('--lr_update_cycle', type=int, default=50000, help='cycle of updating learning rate')
parser.add_argument('--max_epoch', type=int, default=500, help='training epoch')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loader processors')
parser.add_argument('--noise_size', type=int, default=128, help='number of G\'s input')
parser.add_argument('--lambda_adv', type=float, default=34.0, help='adv\'s lambda')
parser.add_argument('--lambda_gp', type=float, default=0.5, help='gp\'s lambda')
parser.add_argument('--model_dump_path', type=str, default='../../resource/gan_models', help='model\'s save path')
parser.add_argument('--verbose', type=bool, default=True, help='output verbose messages')
parser.add_argument('--tmp_path', type=str, default='../../resource/training_temp_1/', help='path of the intermediate files during training')
parser.add_argument('--verbose_T', type=int, default=100, help='steps for saving intermediate file')
parser.add_argument('--logfile', type=str, default='../../resource/training.log', help='logging path')


##########################################
# Load params
#
opt = parser.parse_args()
avatar_tag_dat_path = opt.avatar_tag_dat_path
learning_rate = opt.learning_rate
beta_1 = opt.beta_1
batch_size= opt.batch_size
lr_update_cycle = opt.lr_update_cycle
max_epoch = opt.max_epoch
num_workers= opt.num_workers
noise_size = opt.noise_size
lambda_adv = opt.lambda_adv
lambda_gp = opt.lambda_gp
model_dump_path = opt.model_dump_path
verbose = opt.verbose
tmp_path= opt.tmp_path
verbose_T = opt.verbose_T
logfile = opt.logfile

logger = logging.getLogger()
logger.setLevel(logging.INFO)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log = logging.FileHandler(logfile, mode='w+')
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
log.setFormatter(formatter)

plog = logging.StreamHandler()
plog.setLevel(logging.INFO)
plog.setFormatter(formatter)

logger.addHandler(log)
logger.addHandler(plog)

logger.info('Currently use {} for calculating'.format(device))
if __DEBUG__:
  batch_size = 10
  num_workers = 1
#
#
##########################################


def initital_network_weights(element):
  if hasattr(element, 'weight'):
    element.weight.data.normal_(.0, .02)


def adjust_learning_rate(optimizer, iteration):
  lr = learning_rate * (0.1 ** (iteration // lr_update_cycle))
  return lr


class SRGAN():
  def __init__(self):
    logger.info('Set Data Loader')
    self.dataset = AnimeFaceDataset(avatar_tag_dat_path=avatar_tag_dat_path,
                                    transform=transforms.Compose([ToTensor()]))
    self.data_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers, drop_last=True)
    checkpoint, checkpoint_name = self.load_checkpoint(model_dump_path)
    if checkpoint == None:
      logger.info('Don\'t have pre-trained model. Ignore loading model process.')
      logger.info('Set Generator and Discriminator')
      self.G = Generator().to(device)
      self.D = Discriminator().to(device)
      logger.info('Initialize Weights')
      self.G.apply(initital_network_weights).to(device)
      self.D.apply(initital_network_weights).to(device)
      logger.info('Set Optimizers')
      self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
      self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
      self.epoch = 0
    else:
      logger.info('Load Generator and Discriminator')
      self.G = Generator().to(device)
      self.D = Discriminator().to(device)
      logger.info('Load Pre-Trained Weights From Checkpoint'.format(checkpoint_name))
      self.G.load_state_dict(checkpoint['G'])
      self.D.load_state_dict(checkpoint['D'])
      logger.info('Load Optimizers')
      self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
      self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
      self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
      self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
      self.epoch = checkpoint['epoch']
    logger.info('Set Criterion')
    self.label_criterion = nn.BCEWithLogitsLoss().to(device)
    self.tag_criterion = nn.MultiLabelSoftMarginLoss().to(device)


  def load_checkpoint(self, model_dir):
    models_path = utils.read_newest_model(model_dir)
    if len(models_path) == 0:
      return None, None
    models_path.sort()
    new_model_path = os.path.join(model_dump_path, models_path[-1])
    if torch.cuda.is_available():
      checkpoint = torch.load(new_model_path)
    else:
      checkpoint = torch.load(new_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    return checkpoint, new_model_path


  def train(self):
    iteration = -1
    label = Variable(torch.FloatTensor(batch_size, 1.0)).to(device)
    logging.info('Current epoch: {}. Max epoch: {}.'.format(self.epoch, max_epoch))
    while self.epoch <= max_epoch:
      msg = {}
      adjust_learning_rate(self.optimizer_G, iteration)
      adjust_learning_rate(self.optimizer_D, iteration)
      for i, (avatar_tag, avatar_img) in enumerate(self.data_loader):
        iteration += 1
        if avatar_img.shape[0] != batch_size:
          logging.warn('Batch size not satisfied. Ignoring.')
          continue
        if verbose:
          if iteration % verbose_T == 0:
            msg['epoch'] = int(self.epoch)
            msg['step'] = int(i)
            msg['iteration'] = iteration
        avatar_img = Variable(avatar_img).to(device)
        avatar_tag = Variable(torch.FloatTensor(avatar_tag)).to(device)
        # D : G = 2 : 1
        # 1. Training D
        # 1.1. use really image for discriminating
        self.D.zero_grad()
        label_p, tag_p = self.D(avatar_img)
        label.data.fill_(1.0)

        # 1.2. real image's loss
        real_label_loss = self.label_criterion(label_p, label)
        real_tag_loss = self.tag_criterion(tag_p, avatar_tag)
        real_loss_sum = real_label_loss * lambda_adv / 2.0 + real_tag_loss * lambda_adv / 2.0
        real_loss_sum.backward()
        if verbose:
          if iteration % verbose_T == 0:
            msg['discriminator real loss'] = float(real_loss_sum)

        # 1.3. use fake image for discriminating
        g_noise, fake_tag = utils.fake_generator(batch_size, noise_size, device)
        fake_feat = torch.cat([g_noise, fake_tag], dim=1)
        fake_img = self.G(fake_feat).detach()
        fake_label_p, fake_tag_p = self.D(fake_img)
        label.data.fill_(.0)

        # 1.4. fake image's loss
        fake_label_loss = self.label_criterion(fake_label_p, label)
        fake_tag_loss = self.tag_criterion(fake_tag_p, fake_tag)
        fake_loss_sum = fake_label_loss * lambda_adv / 2.0 + fake_tag_loss * lambda_adv / 2.0
        fake_loss_sum.backward()
        if verbose:
          if iteration % verbose_T == 0:
            msg['discriminator fake loss'] = float(fake_loss_sum)

        # 1.5. gradient penalty
        # https://github.com/jfsantos/dragan-pytorch/blob/master/dragan.py
        alpha_size = [1] * avatar_img.dim()
        alpha_size[0] = avatar_img.size(0)
        alpha = torch.rand(alpha_size).to(device)
        x_hat = Variable(alpha * avatar_img.data + (1 - alpha) * \
                         (avatar_img.data + 0.5 * avatar_img.data.std() * Variable(torch.rand(avatar_img.size())).to(device)),
                         requires_grad=True).to(device)
        pred_hat, pred_tag = self.D(x_hat)
        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].view(x_hat.size(0), -1)
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty.backward()
        if verbose:
          if iteration % verbose_T == 0:
            msg['discriminator gradient penalty'] = float(gradient_penalty)

        # 1.6. update optimizer
        self.optimizer_D.step()

        # 2. Training G
        # 2.1. generate fake image
        self.G.zero_grad()
        g_noise, fake_tag = utils.fake_generator(batch_size, noise_size, device)
        fake_feat = torch.cat([g_noise, fake_tag], dim=1)
        fake_img = self.G(fake_feat)
        fake_label_p, fake_tag_p = self.D(fake_img)
        label.data.fill_(1.0)

        # 2.2. calc loss
        label_loss_g = self.label_criterion(fake_label_p, label)
        tag_loss_g = self.tag_criterion(fake_tag_p, fake_tag)
        loss_g = label_loss_g  * lambda_adv / 2.0 + tag_loss_g * lambda_adv / 2.0
        loss_g.backward()
        if verbose:
          if iteration % verbose_T == 0:
            msg['generator loss'] = float(loss_g)

        # 2.2. update optimizer
        self.optimizer_G.step()

        if verbose:
          if iteration % verbose_T == 0:
            logger.info('------------------------------------------')
            for key in msg.keys():
              logger.info('{} : {}'.format(key, msg[key]))
        # save intermediate file
        if iteration % verbose_T == 0:
          vutils.save_image(avatar_img.data.view(batch_size, 3, avatar_img.size(2), avatar_img.size(3)),
                            os.path.join(tmp_path, 'real_image_{}.png'.format(str(iteration).zfill(8))))
          g_noise, fake_tag = utils.fake_generator(batch_size, noise_size, device)
          fake_feat = torch.cat([g_noise, fake_tag], dim=1)
          fake_img = self.G(fake_feat)
          vutils.save_image(fake_img.data.view(batch_size, 3, avatar_img.size(2), avatar_img.size(3)),
                            os.path.join(tmp_path, 'fake_image_{}.png'.format(str(iteration).zfill(8))))
          logger.info('Saved intermediate file in {}'.format(os.path.join(tmp_path, 'fake_image_{}.png'.format(str(iteration).zfill(8)))))
      # dump checkpoint
      torch.save({
        'epoch': self.epoch,
        'D': self.D.state_dict(),
        'G': self.G.state_dict(),
        'optimizer_D': self.optimizer_D.state_dict(),
        'optimizer_G': self.optimizer_G.state_dict(),
      }, '{}/checkpoint_{}.tar'.format(model_dump_path, str(self.epoch).zfill(4)))
      logger.info('Checkpoint saved in: {}'.format('{}/checkpoint_{}.tar'.format(model_dump_path, str(self.epoch).zfill(4))))
      self.epoch += 1


if __name__ == '__main__':
  if not os.path.exists(model_dump_path):
    os.mkdir(model_dump_path)
  if not os.path.exists(tmp_path):
    os.mkdir(tmp_path)
  gan = SRGAN()
  gan.train()