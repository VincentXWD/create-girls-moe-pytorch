__author__ = 'Wendong Xu'
'''
Training strategy of the `DRAGAN-like SRGAN`.
Have some difference in loss calculating.
I weighted label's loss and tag's loss with half of lambda_adv.
The label_criterion was also different.
'''
import argparse
import networks
from data_loader import AnimeFaceDataset
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
import utils
import random
import os


__DEBUG__ = False

# have GPU or not.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet-GAN")
parser.add_argument('--avatar_tag_dat_path', type=str, default='../../resource/avatar_with_tag.dat', help='avatar with tag\'s list path')

parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.5, help='adam optimizer\'s paramenter')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size for each epoch')
parser.add_argument('--lr_update_cycle', type=int, default=50000, help='cycle of updating learning rate')
parser.add_argument('--max_epoch', type=int, default=1, help='training epoch')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loader processors')
parser.add_argument('--noise_size', type=int, default=128, help='number of G\'s input')
parser.add_argument('--lambda_adv', type=float, default=34.0, help='adv\'s lambda')
parser.add_argument('--lambda_gp', type=float, default=0.5, help='gp\'s lambda')
parser.add_argument('--model_dump_path', type=str, default='../../resource/gan_models', help='model\'s save path')
parser.add_argument('--verbose', type=bool, default=True, help='output verbose messages')


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

if __DEBUG__:
  batch_size = 5
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


def fake_tag():
  tag = [0 for _ in range(len(utils.tag))]
  hair_id = random.randint(0, len(utils.hair)-1)
  eye_id = random.randint(0 ,len(utils.eyes)-1)
  tag[utils.tag_map[utils.hair[hair_id]]] = 1
  tag[utils.tag_map[utils.eyes[eye_id]]] = 1
  for feat in utils.others:
    choice = random.randint(1, 4)
    if choice == 4:
      tag[utils.tag_map[feat]] = 1
  tag = Variable(torch.FloatTensor(tag)).view(1,-1)
  return tag


def fake_generator():
  noise = Variable(torch.FloatTensor(batch_size, noise_size)).to(device)
  noise.data.normal_(.0, .02)
  for _ in range(batch_size):
    tag = torch.cat([fake_tag() for i in range(batch_size)], dim=0)
  return noise, tag


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
    # TODO: fix the criterions
    self.label_criterion = nn.BCEWithLogitsLoss().to(device)
    self.tag_criterion = nn.BCEWithLogitsLoss().to(device)



  def load_checkpoint(self, model_path):
    # TODO:
    pass

  def train(self):
    iteration = -1
    label = Variable(torch.FloatTensor(batch_size, 1.0)).to(device)
    for epoch in range(max_epoch):
      adjust_learning_rate(self.optimizer_G, iteration)
      adjust_learning_rate(self.optimizer_D, iteration)

      for i, (avatar_tag, avatar_img) in enumerate(self.data_loader):
        if verbose:
          if i % 100 == 0:
            msg = {}
            msg['epoch'] = int(epoch)
            msg['step'] = int(i)
        iteration += 1
        avatar_img = Variable(avatar_img).to(device)
        avatar_tag = Variable(torch.FloatTensor(avatar_tag)).to(device)
        # D : G = 1 : 1
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
          if i % 100 == 0:
            msg['discriminator real loss'] = float(real_loss_sum)

        # 1.3. use fake image for discriminating
        g_noise, fake_tag = fake_generator()
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
          if i % 100 == 0:
            msg['discriminator fake loss'] = float(fake_loss_sum)

        # 1.5. gradient penalty
        # https://github.com/jfsantos/dragan-pytorch/blob/master/dragan.py
        alpha_size = [1] * avatar_img.dim()
        alpha_size[0] = avatar_img.size(0)
        alpha = torch.rand(alpha_size).to(device)
        x_hat = Variable(alpha * avatar_img.data + (1 - alpha) * \
                         (avatar_img.data + 0.5 * avatar_img.data.std() * torch.rand(avatar_img.size())),
                         requires_grad=True)
        pred_hat, pred_tag = self.D(x_hat)
        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty.backward()
        if verbose:
          if i % 100 == 0:
            msg['discriminator gradient penalty'] = float(gradient_penalty)

        # 1.6. update optimizer
        self.optimizer_D.step()

        # 2. Training G
        # 2.1. generate fake image
        self.G.zero_grad()
        g_noise, fake_tag = fake_generator()
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
          if i % 100 == 0:
            msg['generator loss'] = float(loss_g)

        # 2.2. update optimizer
        self.optimizer_G.step()

        if verbose:
          if i % 100 == 0:
            print('------------------------------------------')
            for key in msg.keys():
              print('{} : {}'.format(key, msg[key]))
        if __DEBUG__:
          break

      # dump checkpoint
      torch.save({
        'epoch': epoch + 1,
        'D': self.D.state_dict(),
        'G': self.G.state_dict(),
        'optimizer_D': self.optimizer_D.state_dict(),
        'optimizer_G': self.optimizer_G.state_dict(),
      }, '{}/checkpoint_{}.tar'.format(model_dump_path, str(epoch).zfill(4)))


def main():
  if not os.path.exists(model_dump_path):
    os.mkdir(model_dump_path)
  gan = SRGAN()
  gan.train()


if __name__ == '__main__':
  main()