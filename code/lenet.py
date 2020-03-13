# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(1,4,kernel_size=(5,5))
        self.pool1=nn.AvgPool2d((2,2))
        self.conv2=nn.Conv2d(4,16,kernel_size=(5,5))
        self.pool2=nn.AvgPool2d((2,2))
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,26)
    
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool1(x)

        x=F.relu(self.conv2(x))
        x=self.pool2(x)
        
        x=x.view(x.shape[0],16*5*5)
        
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.softmax(self.fc3(x),dim=1)
        
        return x