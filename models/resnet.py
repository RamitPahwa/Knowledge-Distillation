import math
import os

import numpy as np
import torch.nn as nn
import torchvision


def conv3x3(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1,bias=False )


class BasicBlock(nn.Module):

    expansion = 1 

    def __init__(self,inplane,planes,stride=1,downsample=None):
        '''Define Various layers '''

        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(inplane, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU(inplace = True)

        self.conv2 = conv3x3(planes,planes,stride)
        self.bn2 = nn.BatchNorm2D(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(x):
        '''Forward Pass fo the Net '''
        residual = x
        
        out =  self.conv1(x)
        out =  self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out =self.bn2(out)

        if self.downsample is not None:
            out = self.downsample(x)

        out + = residual

        out = self.relu(out)
        return out

class BottleNeck(nn.Module):
    
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(BottleNeck,self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, stride = stride, kernel_size = 1 ,bias = False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(planes, planes, stride=stride, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, stride=stride, kernel_size=3,bias=False)
        self.bn3 = nn.Conv2d(planes*self..expansion)
        self.downsample = downsample

        self.stride = stride

    def forward(x):
        
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            out = self.downsample(out)
        out+=residual
        out = self.relu(out)
        return out 





