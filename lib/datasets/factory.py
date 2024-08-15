# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.clipart import clipart
from datasets.sketch import sketch
from datasets.watercolor import watercolor
from datasets.bdd import bdd
from datasets.sim10k import sim10k
from datasets.kitti import kitti
from datasets.cityscapes import cityscapes

import numpy as np


###########################################
for year in ['2007']:
  for split in ['trainval','train', 'val', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for split in ['trainval','train','val','test']:
  name = 'clipart22_{}'.format(split)
  __sets[name] = (lambda split=split : clipart(split))

for split in ['trainval','train','val','test']:
  name = 'sketch22_{}'.format(split)
  __sets[name] = (lambda split=split : sketch(split))

for split in ['trainval','train','val','test']:
  name = 'watercolor22_{}'.format(split)
  __sets[name] = (lambda split=split : watercolor(split))

for split in ['clear', 'dark', 'foggy', 'rainy']:
  name = split
  __sets[name] = (lambda split=split: bdd(split))


###########################################


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
