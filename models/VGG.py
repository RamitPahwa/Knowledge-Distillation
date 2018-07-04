import torch.nn as nn
import numpy as np 
import torch.nn.functional as F


class VGG(nn.Module):
    ''' Implementation of VGG Network to distill the output '''

    def __init__(self,params):
        super(VGG,self).__init__():
        ''' Feature Extraction Layers 
            Each convolution layer has arguments : (input_channels, output_channels, kernel_size, stride, padding)
        '''
        # self.channels = params.channels
        self.channels = 32 
        self.out_channels = 10
        # self.dropout_rate = params.dropout_rate
        self.dropout_rate = 0.5

        self.conv1 = nn.Conv2d(3, self.channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.conv2 = nn.Conv2d(self.channels, self.channels*2, kernel_size=3, stride=1, padding=1)
        self.bn2  = nn.BatchNorm2d(self.channels*2)
        self.conv3 = nn.Conv2d(self.channels*2, self.channels*4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.channels*4)

        ''' Classifier Layer 
            Downsample to the number of target classes 
        '''

        self.fc1 = nn.Linear(self.channels*4, self.channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.channels*4)
        self.fc2 = nn.Linear(self.channels*4, self.out_channels)

    def forward(self, x):

        ''' Forward Pass of the network , Functional Calls '''
        # input : batch_size x 3 x 32 x 32
        x = self.conv1(x)               # batch_size x num_channels x 32 x 32
        x = self.bn1(x)                 # batch_size x num_channels x 32 x 32
        x = F.relu(F.max_poold2d(x, 2)) # batch_size x num_channels x 16 x 16
        x = self.conv2(x)               # batch_size x num_channels*2 x 16 x 16
        x = self.bn2(x)
        x = F.relu(F.max_pool2d(x, 2))  # batch_size x num_channels*2 x 8 x 8 
        x = self.conv3(x)               # batch_size x num_channels*4 x 8 x 8 
        x = self.bn3(x)
        x = F.relu(F.max_pool2d(x, 2))  # batch_size x num_channels*4 x 4 x 4

        # flatten the output to input 

        x = x.view(-1, 4*4*self.num_channels*4) # batch_size x 4*4*num_channels*4

        x = F.dropout(F.relu(self.fcbn1(self.fc1(x))), p = self.dropout_rate, training = self.training)  # batch_size x num_channels*4 
        x = self.fc2(x)                                                                                  # batch_size x 10

        return x

def loss_function(outputs, targets):

    ''' Compute cross entropy loss between ouputs and targets 
        Pytorch has various loss function defined 
    '''

    return nn.CrossEntropyLoss()(outputs, targets)

def loss_function_kd(outputs, targets, teacher_outputs, params):

    ''' Computes the Knowledge Distillation Loss given outputs, actual targets and teacher_ouputs

        KL divergence in Pytorch require the input tensor to be log probabilities
        labels should probabilities and output should be log probabilities, as log_softmax is more efficient than taking softmax then log 
        Hyperparamter are alpha and temperature
    
    '''
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim =1), 
                            F.softmax(teacher_outputs/T, dim =1 )*(alpha * T * T) + \
                            F.cross_entropy(outputs,targets) * (1 - aplha))
    return KD_loss

def accuracy(outputs, targets):
    ''' Compute the accuracy , given the ouputs and labels '''

    outputs = np.argmax(output, axis = 1)
    return np.sum(outputs == targets)/float(targets.size)







        
        


