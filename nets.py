import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as Func
from   torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from os.path import join

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=[5,5],stride=1,padding=2)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,64,[5,5],padding=2)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,128,[5,5],padding=2)
        self.conv3_bn = nn.BatchNorm2d(128)
    
        self.conv3u = nn.Conv2d(128,128,[5,5],padding=2)
        self.deconv3 = nn.ConvTranspose2d(128,64,kernel_size = 2,stride = [2,2])
        self.deconv3_bn = nn.BatchNorm2d(64)

        self.conv2u = nn.Conv2d(64,64,[5,5],padding=2)
        self.deconv2 = nn.ConvTranspose2d(64,32,kernel_size = 2,stride = [2,2])
        self.deconv2_bn = nn.BatchNorm2d(32)

        self.conv1u = nn.Conv2d(32,32,[5,5],padding=2)
        self.deconv1 = nn.ConvTranspose2d(32,1,kernel_size = 2,stride = [2,2])        

        pass

    def forward(self, x):
        input = x #50,1,64,64
        conv1 = Func.max_pool2d(Func.leaky_relu(self.conv1_bn(self.conv1(x))),kernel_size = 2, stride=[2,2]) #50,32,32,32
        conv2 = Func.max_pool2d(Func.leaky_relu(self.conv2_bn(self.conv2(conv1))),kernel_size = 2, stride=[2,2]) #50,64,16,16
        conv3 = Func.max_pool2d(Func.leaky_relu(self.conv3_bn(self.conv3(conv2))),kernel_size = 2, stride=[2,2]) #50,128,8,8        

        #x = torch.cat((x,conv2),1)
        x = Func.leaky_relu(self.conv3_bn(self.conv3u(conv3))) #50,128,8,8
        x = Func.leaky_relu(self.deconv3_bn(self.deconv3(x))) #50,64,16,16

        x = Func.leaky_relu(self.conv2_bn(self.conv2u(x))) #50,64,16,16
        x = Func.leaky_relu(self.deconv2_bn(self.deconv2(x))) #50,32,32,32
        
        x = Func.leaky_relu(self.conv1_bn(self.conv1u(x))) #50,32,32,32
        logits = self.deconv1(x) #50,1,64,64

        return Func.sigmoid(logits)