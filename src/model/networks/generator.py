__author__ = 'Wendong Xu'
'''
Use PReLU instead of ReLU.
intput: tag's one-hot map
output: image(NCWH)
'''

import torch.nn as nn


class _ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, bias=False):
    super(_ResidualBlock, self).__init__()
    self.conv_1 = nn.Conv2d(in_channels, out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.bn_1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.PReLU()
    self.conv_2 = nn.Conv2d(out_channels, out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.bn_2 = nn.BatchNorm2d(out_channels)

  def forward(self, tensor):
    r_tensor = tensor
    output = self.conv_1(tensor)
    output = self.bn_1(output)
    output = self.relu(output)
    output = self.conv_2(output)
    output = self.bn_2(output)
    output += r_tensor
    return output


class _SubpixelBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, bias=False, upscale_factor=2):
    super(_SubpixelBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels,
                 kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    self.bn = nn.BatchNorm2d(in_channels)
    self.relu = nn.PReLU()

  def forward(self, tensor):
    output = self.conv(tensor)
    output = self.pixel_shuffle(output)
    output = self.bn(output)
    output = self.relu(output)
    return output


class Generator(nn.Module):
  def __init__(self, tag=34):
    super(Generator, self).__init__()
    in_channels = 128 + tag
    self.dense_1 = nn.Linear(in_channels, 64*16*16)
    self.bn_1 = nn.BatchNorm2d(64)
    self.relu_1 = nn.PReLU()
    self.residual_layer = self.make_residual_layer(16) # outn=64
    self.bn_2 = nn.BatchNorm2d(64)
    self.relu_2 = nn.PReLU()
    self.subpixel_layer = self.make_subpixel_layer(3) # outn=64
    self.conv_1 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=True)
    self.tanh_1 = nn.Tanh()

  def forward(self, tensor):
    output = self.dense_1(tensor)
    output = output.view(-1, 64, 16, 16)
    output = self.bn_1(output)
    output = self.relu_1(output)
    r_output = output
    output = self.residual_layer(output)
    output = self.bn_2(output)
    output = self.relu_2(output)
    output += r_output
    output = self.subpixel_layer(output)
    output = self.conv_1(output)
    output = self.tanh_1(output)
    return output

  def make_residual_layer(self, block_size=16):
    layers = []
    for _ in range(block_size):
      layers.append(_ResidualBlock(64, 64, 3, 1))
    return nn.Sequential(*layers)

  def make_subpixel_layer(self, block_size=3):
    layers = []
    for _ in range(block_size):
      layers.append(_SubpixelBlock(64, 256, 3, 1))
    return nn.Sequential(*layers)


if __name__ == '__main__':
  from torch.autograd import Variable
  import torch

  gen = Generator()
  x = Variable(torch.rand((1, 128+34)), requires_grad=True)
  print(gen(x).shape)

