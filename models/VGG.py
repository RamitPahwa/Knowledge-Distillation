import torch.nn as nn
import numpy as np 
import torch.nn.functional as F


class VGG(nn.Module):
    ''' Implementation of VGG Network to distill the output '''

    def __init__(self):
        super(VGG,self).__init__():
        ''' Feature Extraction Layers 
            Each convolution layer has arguments : (input_channels, output_channels, kernel_size, stride, padding)
            Experiment on padding and stride?
        '''

        self.channels = 32 
        self.out_channels = 10
        self.dropout_rate = 0.5

        self.conv1 = nn.Conv2d(3, self.channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.conv2 = nn.Conv2d(self.channels, self.channels*2, kernel_size=3, stride=1, padding=1)
        self.bn2  = nn.BatchNorm2d(self.channels*2)
        self.conv3 = nn.Conv2d(self.channels*2, self.channels*4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.channels*4)

        ''' Classifier Layer 
            First Layer is Linear : Replace with AutoEncoder ?
            Downsample to the number of target classes 
        '''

        self.fc1 = nn.Linear(self.channels*4, self.channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.channels*4)
        self.fc2 = nn.Linear(self.channels*4, self.out_channels)

    def forward(self, x):

        ''' Forward Pass of the network , Functional Calls '''

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(F.max_poold2d(x, 2))
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(F.max_pool2d(x, 2))

        




        
        


