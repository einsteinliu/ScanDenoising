import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from   torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import prepare_data
import numpy as np

class DocCleanNet(nn.Module):
    def __init__(self):
        super(DocCleanNet,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=[5,5],stride=1,padding=2)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,[5,5],padding=2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.dense1 = nn.Linear(7*7*64, 7*7*128)
        self.dense2 = nn.Linear(7*7*128, 7*7*128)
        self.conv21 = nn.Conv2d(128,64,5,padding=2)
        self.conv21_bn = nn.BatchNorm2d(64)
        self.deconv21 = nn.ConvTranspose2d(64,32,kernel_size = 2,stride = [2,2])
        self.deconv21_bn = nn.BatchNorm2d(32)
        self.conv22 = nn.Conv2d(32,32,5,padding=2)        
        self.conv22_bn = nn.BatchNorm2d(32)
        self.deconv22 = nn.ConvTranspose2d(32,1,kernel_size = 2,stride = [2,2])
        pass

    def forward(self, x):
        x = Func.max_pool2d(Func.leaky_relu(self.conv1_bn(self.conv1(x))),kernel_size = 2, stride=[2,2])
        x = Func.max_pool2d(Func.leaky_relu(self.conv2_bn(self.conv2(x))),kernel_size = 2, stride=[2,2])
        x = x.view(-1,7*7*64)
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.view(-1,128,7,7)
        x = Func.leaky_relu(self.conv21_bn(self.conv21(x)))
        x = Func.leaky_relu(self.deconv21_bn(self.deconv21(x)))
        x = Func.leaky_relu(self.conv22_bn(self.conv22(x)))
        logits = self.deconv22(x)
        return Func.sigmoid(logits)
       
class DocDataset(Dataset):
    def __init__(self, train = True, source = "./train/",target = "./train_cleaned/"):
        self.data_set = prepare_data.prepare_data_set(source, target)
        self.train_image_dirty = np.float32(np.asarray(self.data_set["dirty"])) / 255.0
        self.train_image_clean = np.float32(np.asarray(self.data_set["clean"])) / 255.0
        self.transform = transforms.transforms.ToTensor()
    def __getitem__(self,index):
        return self.train_image_dirty[index],self.train_image_clean[index]
    def __len__(self):
        return len(self.train_image_dirty)

dataset = DocDataset()
loader = DataLoader(dataset,batch_size=50,shuffle=True)

denoise_net = DocCleanNet()
denoise_net.cuda()
optimizer = optim.Adam(denoise_net.parameters(),lr=0.001)

for epoch in range(1000):
    denoise_net.train()
    for batch_id,[dirty_batch,clean_batch] in enumerate(loader):
        dirty = Variable(dirty_batch.view(-1,1,28,28).cuda())
        clean = Variable(clean_batch.view(-1,1,28,28).cuda())
        optimizer.zero_grad()
        out = denoise_net(dirty)
        loss = Func.l1_loss(out,clean)
        if(batch_id%100==0):
            print(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()
    

