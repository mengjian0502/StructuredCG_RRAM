"""
ResNet on CIFAR10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from .cg import Conv2d_CG, QConv2d_CG

class DownsampleA(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, stride=1, downsample=None, slice_size=16, cg_groups=4, cg_threshold_init=0, cg_alpha=2, hard_sig=False, wbit=32, abit=32, mtype=5):
    super(ResNetBasicblock, self).__init__()   
    self.conv_a = QConv2d_CG(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, 
                            slice_size=slice_size, cg_groups=cg_groups, cg_threshold_init=cg_threshold_init, 
                            cg_alpha=cg_alpha, hard_sig=hard_sig, wbit=wbit, abit=abit, mtype=mtype)
    self.bn_a = nn.BatchNorm2d(planes)
    self.relu1 = nn.ReLU(inplace=True)
 
    self.conv_b = QConv2d_CG(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, 
                            slice_size=slice_size, cg_groups=cg_groups, cg_threshold_init=cg_threshold_init, 
                            cg_alpha=cg_alpha, hard_sig=hard_sig, wbit=wbit, abit=abit, mtype=mtype)
    self.bn_b = nn.BatchNorm2d(planes)
    self.relu2 = nn.ReLU(inplace=True)
    self.downsample = downsample

  def forward(self, x):
    residual = x

    basicblock = self.conv_a(x)
    basicblock = self.bn_a(basicblock)
    basicblock = self.relu1(basicblock)

    basicblock = self.conv_b(basicblock)
    basicblock = self.bn_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    basicblock = residual + basicblock
    basicblock = self.relu2(basicblock)
    return basicblock


class CifarResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, depth, num_classes, slice_size=16, cg_groups=1, cg_threshold_init=0, cg_alpha=2, hard_sig=False, wbit=32, abit=32, mtype=5):
    super(CifarResNet, self).__init__()
    block = ResNetBasicblock

    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))
    self.num_classes = num_classes
    self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
  
    self.relu0 = nn.ReLU(inplace=True)
    self.bn_1 = nn.BatchNorm2d(16)
    
    self.inplanes = 16
    self.stage_1 = self._make_layer(block, 16, layer_blocks, 1, slice_size=slice_size, 
                  cg_groups=cg_groups, cg_threshold_init=cg_threshold_init, cg_alpha=cg_alpha, hard_sig=hard_sig, wbit=wbit, abit=abit, mtype=mtype)
    self.stage_2 = self._make_layer(block, 32, layer_blocks, 2, slice_size=slice_size, 
                  cg_groups=cg_groups, cg_threshold_init=cg_threshold_init, cg_alpha=cg_alpha, hard_sig=hard_sig, wbit=wbit, abit=abit, mtype=mtype)
    self.stage_3 = self._make_layer(block, 64, layer_blocks, 2, slice_size=slice_size, 
                  cg_groups=cg_groups, cg_threshold_init=cg_threshold_init, cg_alpha=cg_alpha, hard_sig=hard_sig, wbit=wbit, abit=abit, mtype=mtype)
    self.avgpool = nn.AvgPool2d(8)
    self.classifier = nn.Linear(64*block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, slice_size=16, cg_groups=1, cg_threshold_init=0, cg_alpha=2, hard_sig=False, wbit=32, abit=32, mtype=5):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          QConv2d_CG(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, 
                    slice_size=slice_size, cg_groups=1, cg_threshold_init=cg_threshold_init, cg_alpha=cg_alpha, hard_sig=hard_sig, wbit=wbit, abit=abit, mtype=mtype),
          nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, slice_size=slice_size, cg_groups=cg_groups, 
                    cg_threshold_init=cg_threshold_init, cg_alpha=cg_alpha, hard_sig=hard_sig, wbit=wbit, abit=abit, mtype=mtype))
    self.inplanes = planes * block.expansion
    
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, slice_size=slice_size, 
                        cg_groups=cg_groups, cg_threshold_init=cg_threshold_init, cg_alpha=cg_alpha, hard_sig=hard_sig, wbit=wbit, abit=abit, mtype=mtype))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv_1_3x3(x)
    x = self.relu0(self.bn_1(x))
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
  
  def get_global_mp(self):
    val = torch.Tensor()
    if torch.cuda.is_available():
        val = val.cuda()
    
    count = 0
    for m in self.modules():
      if isinstance(m, QConv2d_CG):
        if not count in [0]:
          w_l = m.weight
          w_l = w_l.permute(0,2,3,1)
          num_group = w_l.size(0)*w_l.size(1)*w_l.size(2)
          w_l = w_l.contiguous().view((num_group, w_l.size(3)))

          g = w_l.abs().mean(dim=1)
          val = torch.cat((val.view(-1), g.view(-1)))
        count += 1
    return val

  
  def get_global_mp_thre(self, ratio):
    grp_val = self.get_global_mp()
    sorted_block_values, indices = torch.sort(grp_val.contiguous().view(-1))
    thre_index = int(grp_val.data.numel() * ratio)
    threshold = sorted_block_values[thre_index]
    return threshold

  def update_mtype(self, v):
    for m in self.modules():
      if isinstance(m, QConv2d_CG):
        m.mtype = v


class resnet20_CG:
  base=CifarResNet 
  args = list()
  kwargs = {'depth': 20}

class resnet32_CG:
  base=CifarResNet
  args = list()
  kwargs = {'depth': 32}
