__author__ = 'Wendong Xu'
import argparse
import networks
from data_loader import AnimeFaceDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
import numpy as np


__DEBUG__ = True

# has GPU or not.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet-GAN")
parser.add_argument('--avatar_tag_dat_path', type=str, default='../../resource/avatar_with_tag.dat', help='avatar with tag\'s list path')
parser.add_argument('--data_augmentation', type=bool, default=True, help='need data augmentation or not')

parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.5, help='adam optimizer\'s paramenter')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size for each epoch')
parser.add_argument('--lr_update_cycle', type=int, default=50000, help='cycle of updating learning rate')
parser.add_argument('--max_epoch', type=int, default=1, help='training epoch')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loader processors')


##########################################
# Load params
#
opt = parser.parse_args()

avatar_tag_dat_path = opt.avatar_tag_dat_path
data_augmentation = opt.data_augmentation
learning_rate = opt.learning_rate
beta_1 = opt.beta_1
batch_size= opt.batch_size
lr_update_cycle = opt.lr_update_cycle
max_epoch = opt.max_epoch
num_workers= opt.num_workers
if __DEBUG__:
  batch_size = 1
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
    # Set Data Loader
    self.dataset = AnimeFaceDataset(avatar_tag_dat_path=avatar_tag_dat_path,
                                        transform=transforms.Compose([ToTensor()]))
    self.data_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers, drop_last=True)
    # Set Generator and Discriminator
    self.G = networks.Generator().to(device)
    self.D = networks.Discriminator().to(device)

    # Initialize weights
    self.G.apply(initital_network_weights).to(device)
    self.D.apply(initital_network_weights).to(device)

    # Set Optimizers
    self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
    self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=learning_rate, betas=(beta_1, 0.999))

    # Set Criterion
    # TODO: fix the criterion
    self.criterion = nn.BCEWithLogitsLoss().to(device)
    self.tag_criterion = nn.BCEWithLogitsLoss().to(device)

  def train(self):
    iteration = 0
    label = Variable(torch.FloatTensor(batch_size, 1.0)).to(device)
    for epoch in range(max_epoch):
      adjust_learning_rate(self.optimizer_G, iteration)
      adjust_learning_rate(self.optimizer_D, iteration)

      for i, (avatar_tag, avatar_img) in enumerate(self.data_loader):
        # avatar_img = augmentation(avatar_img)
        avatar_img = Variable(avatar_img).to(device)
        avatar_tag = Variable(torch.FloatTensor(avatar_tag)).to(device)
        ########################################################
        # Training D
        #
        self.D.zero_grad()
        # 1. Use really image for discriminating
        label_p, tag_p = self.D(avatar_img)
        label.data.fill_(1.0)
        # 1.1. Calc L(D)
        real_label_loss = self.criterion(label, label_p)
        real_tag_loss = self.tag_criterion(avatar_tag, tag_p)
        real_loss_sum = real_label_loss + real_tag_loss
        real_loss_sum.backward()

        # 2. Use fake image for discriminating

        #
        #
        ########################################################
        if __DEBUG__:
          break


def main():
  gan = SRGAN()
  gan.train()


if __name__ == '__main__':
  main()