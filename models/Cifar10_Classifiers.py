# Pretrained Models from
# https://github.com/huyvnphan/PyTorch_CIFAR10

import torch
from torch import nn


from sys import path
import os
p = os.path.abspath("")
root_dir = "ATAC_GAN" # Root directory of project
p = p[:p.rindex(root_dir)+len(root_dir)]
if p not in path:
    path.append(p)

# Directory has models and saved weights from https://github.com/huyvnphan/PyTorch_CIFAR10
# These models expect images in range [0,1] and
# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2023, 0.1994, 0.2010]
from models.saves.cifar10_models import *




#------------------
# Model Definitions
#------------------

class Cifar10_Classifier_Factory(object):
    #--------------------------
    # Model Selection Functions
    #--------------------------

    MODELS = {
        'vgg11_bn': vgg11_bn,
        'vgg13_bn': vgg13_bn,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'densenet121': densenet121,
        'densenet161': densenet161,
        'densenet169': densenet169,
        'mobilenet_v2': mobilenet_v2,
        'googlenet': googlenet,
        'inception_v3': inception_v3
    }
    
    @classmethod
    def supported_models(cls):
        return cls.MODELS.keys()
    
    @classmethod
    def get_model(cls, name):
        return cls.MODELS[name]
    