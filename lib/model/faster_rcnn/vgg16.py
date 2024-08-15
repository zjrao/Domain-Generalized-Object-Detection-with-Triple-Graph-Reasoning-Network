# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb
from model.GCN.MDGRL import PLGRL,ILGRL,CDGRL
import numpy as np
import scipy.sparse as sp
import math

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:10])  #128
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False
    self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[10:17])  #256
    self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[17:24])  #512
    self.RCNN_base3 = nn.Sequential(*list(vgg.features._modules.values())[24:-1])   #512

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    self.PLGRL = PLGRL()
    self.ILGRL = ILGRL()
    self.CDGRL = CDGRL()

    self.domain_global = netD()
    self.domain_ins = netD_da()
    self.c_model, self.cp_model = aux_models(4096, 3, 21, layers_cls=[1024, 256])
    self.mixstyle = MixStyle()

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

  def _compute_cls_loss(self, model, feature, label, domain, mode="self"):
    if model is not None:
      feature_list = []
      label_list = []
      weight_list = []
      for i in range(3):
        if mode == "self":
          feature_list.append(feature[domain == i])
          label_list.append(label[domain == i])
        else:
          feature_list.append(feature[domain != i])
          label_list.append(label[domain != i])
        weight = torch.zeros(21).to('cuda')
        for j in range(21):
          weight[j] = 0 if (label_list[-1] == j).sum() == 0 else 1.0 / (label_list[-1] == j).sum().float()
        weight = weight / weight.sum()
        weight_list.append(weight)
      class_logit = model(feature_list)
      loss = 0
      for p, l, w in zip(class_logit, label_list, weight_list):
        if p is None:
          continue
        loss += F.cross_entropy(p, l, weight=w) / 3
    else:
      loss = torch.zeros(1, requires_grad=True).to('cuda')

    return loss


def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

class netD(nn.Module):
  def __init__(self):
    super(netD, self).__init__()
    self.conv1 = conv3x3(512, 512, stride=2)
    self.bn1 = nn.BatchNorm2d(512)
    self.conv2 = conv3x3(512, 128, stride=2)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = conv3x3(128, 128, stride=2)
    self.bn3 = nn.BatchNorm2d(128)
    self.fc = nn.Linear(128, 3)

  def forward(self, x):
    x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
    x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
    x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
    x = F.avg_pool2d(x, (x.size(2), x.size(3)))
    x = x.view(-1, 128)
    x = self.fc(x)
    return x

class netD_da(nn.Module):
  def __init__(self):
    super(netD_da, self).__init__()
    self.fc1 = nn.Linear(4096, 100)
    self.bn1 = nn.BatchNorm1d(100)
    self.fc2 = nn.Linear(100, 100)
    self.bn2 = nn.BatchNorm1d(100)
    self.fc3 = nn.Linear(100, 3)

  def forward(self, x):
    x = F.dropout(F.relu(self.bn1(self.fc1(x))), training=self.training)
    x = F.dropout(F.relu(self.bn2(self.fc2(x))), training=self.training)
    x = self.fc3(x)
    return x

class GradReverse(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, lambd, reverse=True):
    ctx.lambd = lambd
    ctx.reverse = reverse
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    if ctx.reverse:
      return (grad_output * -ctx.lambd), None, None
    else:
      return (grad_output * ctx.lambd), None, None


def grad_reverse(x, lambd=1.0, reverse=True):
  return GradReverse.apply(x, lambd, reverse)

class ClsNet(nn.Module):
  def __init__(self, in_channels, num_domains, num_classes, reverse=True, layers=[1024, 256]):
    super(ClsNet, self).__init__()
    self.classifier_list = nn.ModuleList()
    for _ in range(num_domains):
      class_list = nn.ModuleList()
      class_list.append(nn.Linear(in_channels, layers[0]))
      for i in range(1, len(layers)):
        class_list.append(nn.Sequential(
          nn.ReLU(inplace=True),
          nn.Linear(layers[i - 1], layers[i])
        ))
      class_list.append(nn.ReLU(inplace=True))
      class_list.append(nn.Dropout())
      class_list.append(nn.Linear(layers[-1], num_classes))
      self.classifier_list.append(nn.Sequential(*class_list))
    for m in self.classifier_list.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, .1)
        nn.init.constant_(m.bias, 0.)

    self.lambda_ = 0
    self.reverse = reverse

  def set_lambda(self, lambda_):
    self.lambda_ = lambda_

  def forward(self, x):
    output = []
    for c, x_ in zip(self.classifier_list, x):
      if len(x_) == 0:
        output.append(None)
      else:
        x_ = grad_reverse(x_, self.lambda_, self.reverse)
        output.append(c(x_))

    return output

  def get_params(self, lr):
    return [{"params": self.classifier_list.parameters(), "lr": lr}]


def aux_models(in_channels, num_domains, num_classes, layers_cls=[1024, 256]):
  c_model = ClsNet(in_channels, num_domains, num_classes, reverse=False, layers=layers_cls)
  cp_model = ClsNet(in_channels, num_domains, num_classes, reverse=True, layers=layers_cls)

  return c_model, cp_model

import random
import numpy as np
class MixStyle(nn.Module):
  """MixStyle.
  Reference:
    Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
  """

  def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
    """
    Args:
      p (float): probability of using MixStyle.
      alpha (float): parameter of the Beta distribution.
      eps (float): scaling parameter to avoid numerical issues.
      mix (str): how to mix.
    """
    super().__init__()
    self.p = p
    self.beta = torch.distributions.Beta(alpha, alpha)
    self.eps = eps
    self.alpha = alpha
    self.mix = mix
    self._activated = True

  def __repr__(self):
    return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

  def set_activation_status(self, status=True):
    self._activated = status

  def update_mix_method(self, mix='random'):
    self.mix = mix

  def forward(self, x):
    if not self.training or not self._activated:
      return x

    if random.random() > self.p:
      return x

    B = x.size(0)

    x_copy = x.clone().detach().cpu().numpy()
    mu = torch.from_numpy(np.mean(x_copy, axis=(2,3), keepdims=True)).cuda()
    var = torch.from_numpy(np.var(x_copy, axis=(2,3), keepdims=True)).cuda()
    sig = (var + self.eps).sqrt()
    x_normed = (x - mu) / sig

    lmda = self.beta.sample((B, 1, 1, 1))
    lmda = lmda.to(x.device)

    if self.mix == 'random':
      # random shuffle
      perm = torch.randperm(B)

    elif self.mix == 'crossdomain':
      # split into two halves and swap the order
      perm = torch.arange(B - 1, -1, -1)  # inverse index
      perm_b, perm_a = perm.chunk(2)
      perm_b = perm_b[torch.randperm(B // 2)]
      perm_a = perm_a[torch.randperm(B // 2)]
      perm = torch.cat([perm_b, perm_a], 0)

    else:
      raise NotImplementedError

    mu2, sig2 = mu[perm, :, :, :], sig[perm, :, :, :]
    mu_mix = mu * lmda + mu2 * (1 - lmda)
    sig_mix = sig * lmda + sig2 * (1 - lmda)

    return x_normed * sig_mix + mu_mix