__author__ = 'Wendong Xu'
'''
Without BN so set bias=True.
Remove the Sigmoid before output pred logits.
input: images(NCWH)
output: probability of reality or fake; each tag's probability.
'''

import torch.nn as nn


class _ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, bias=True):
    super(_ResidualBlock, self).__init__()
    self.conv_1 = nn.Conv2d(in_channels, out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.leaky_relu_1 = nn.LeakyReLU()
    self.conv_2 = nn.Conv2d(out_channels, out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.leaky_relu_2 = nn.LeakyReLU()

  def forward(self, tensor):
    r_tensor = tensor
    output = self.conv_1(tensor)
    output = self.leaky_relu_1(output)
    output = self.conv_2(output)
    output += r_tensor
    output = self.leaky_relu_2(output)
    return output


class _Block(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, conv_stride=2, conv_kernel_size=4):
    super(_Block, self).__init__()
    self.conv_1 = nn.Conv2d(in_channels, out_channels,
                    kernel_size=conv_kernel_size, stride=conv_stride, padding=padding, bias=bias)
    self.leaky_relu_1 = nn.LeakyReLU()
    self.residual_block_1 = _ResidualBlock(out_channels, out_channels, kernel_size, stride)
    self.residual_block_2 = _ResidualBlock(out_channels, out_channels, kernel_size, stride)

  def forward(self, tensor):
    output = self.conv_1(tensor)
    output = self.leaky_relu_1(output)
    output = self.residual_block_1(output)
    output = self.residual_block_2(output)
    return output


class Discriminator(nn.Module):
  def __init__(self, tag=34):
    super(Discriminator, self).__init__()
    self.reduce_block_1 = _Block(3, 32, conv_kernel_size=4)
    self.reduce_block_2 = _Block(32, 64, conv_kernel_size=4)
    self.reduce_block_3 = _Block(64, 128, conv_kernel_size=4)
    self.reduce_block_4 = _Block(128, 256, conv_kernel_size=3)
    self.reduce_block_5 = _Block(256, 512, conv_kernel_size=3)
    self.conv_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=True)
    self.leaky_relu_1 = nn.LeakyReLU()
    self.dense_1 = nn.Linear(2*2*1024, 1)
    self.dense_tag = nn.Linear(2*2*1024, tag)
    # self.sigmoid = nn.Sigmoid()

  def forward(self, tensor):
    output = self.reduce_block_1(tensor)
    output = self.reduce_block_2(output)
    output = self.reduce_block_3(output)
    output = self.reduce_block_4(output)
    output = self.reduce_block_5(output)
    output = self.conv_1(output)
    output = self.leaky_relu_1(output)
    output = output.view(output.size(0), -1)
    output1 = self.dense_1(output)
    # output1 = self.sigmoid(output1)
    output2 = self.dense_tag(output)
    # output2 = self.sigmoid(output2)
    return output1, output2


if __name__ == '__main__':
  from torch.autograd import Variable
  import torch

  dis = Discriminator()
  x = Variable(torch.rand((1, 3, 128, 128)), requires_grad=True)
  print(dis(x))

