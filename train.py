import torch
import torch.nn.functional as Func
import torch.optim as optim
from   torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import cv2
import os
import configparser
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import nets
import prepare_data

# Training settings
cmdparser = argparse.ArgumentParser(description='PyTorch Corner')
cmdparser.add_argument('--config', type=str, default="config.ini")
opt = cmdparser.parse_args()
print(opt)

parser = configparser.ConfigParser()
parser.read(opt.config)

source_folder = parser['train']['source_folder']
target_folder = parser['train']['target_folder']
batch_size = int(parser['train']['batch_size'])
model_file = parser['train']['model_file']
loss_file = parser['train']['loss_file']
learning_rate = float(parser['train']['learning_rate'])
maximum_iteration = int(parser['train']['maximum_iteration'])
stop_epsilon = float(parser['train']['stop_epsilon'])
image_width = int(parser['train']['image_width'])
image_height = int(parser['train']['image_height'])
saving_interval = int(parser['train']['saving_interval'])
model_saving_interval = int(parser['train']['model_saving_interval'])

def adaptImageToNet(origin_image, div):
    height = int(origin_image.shape[0]/div)*div
    width = int(origin_image.shape[1]/div)*div
    origin_image = origin_image[:height,:width]
    return np.float32(np.asarray(origin_image))/255.0  

def load_data(folder_src="./train",folder_dst="./train_cleaned"):
    dirty_files = os.listdir(folder_src)
    clean_files = os.listdir(folder_dst)
    dataset = {'dirty':[],'clean':[]}
    if len(dirty_files) == len(clean_files):
        file_num = len(dirty_files)
        for i in range(0,file_num):
            curr_image = cv2.imread(join(folder_src,dirty_files[i]))
            curr_clean = cv2.imread(join(folder_dst,clean_files[i]))
            dataset['dirty'].append(adaptImageToNet(curr_image,8))
            dataset['clean'].append(adaptImageToNet(curr_clean,8))
    return dataset             

class DocDataset(Dataset):
    def __init__(self, train = True, source = "./train/",target = "./train_cleaned/", image_size = 64):
        self.data_set = prepare_data.prepare_data_set(source, target,size=image_size)
        self.train_image_dirty = np.float32(np.asarray(self.data_set["dirty"])) / 255.0
        self.train_image_clean = np.float32(np.asarray(self.data_set["clean"])) / 255.0
        self.transform = transforms.transforms.ToTensor()
    def __getitem__(self,index):
        return self.train_image_dirty[index],self.train_image_clean[index]
    def __len__(self):
        return len(self.train_image_dirty)

dataset = DocDataset(source=source_folder,target=target_folder)
loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

if os.path.exists(model_file):
    denoise_net = torch.load(model_file)
else:
    denoise_net = nets.SegNet()    

denoise_net.cuda()

record = open(loss_file,"w")
min_loss = 99999
batch_count = 0

optimizer = optim.Adam(denoise_net.parameters(),lr=learning_rate)
denoise_net.train()
while(True):
    for batch_id,[source_batch,target_batch] in enumerate(loader):
        batch_count = batch_count + 1
        source = Variable(source_batch.view(-1,1,image_width,image_height).cuda())
        target = Variable(target_batch.view(-1,1,image_width,image_height).cuda())
        optimizer.zero_grad()
        out = denoise_net(source)
        #loss_func = torch.nn.BCEWithLogitsLoss(weight=(target+1.0))
        loss_func = torch.nn.BCELoss()
        loss = loss_func(out,target)

        if batch_count%saving_interval == 0:
            prob = Func.sigmoid(out).data.cpu().numpy()        
            cv2.imwrite(str(batch_count)+'.png',prob[0,0,:,:]*255)                           

        if batch_count%100==0:                
            print(str(batch_count)+':'+str(loss.item())+"\n")
        record.write(str(loss.item())+"\n")
        
        if batch_count%model_saving_interval == 0:                
            torch.save(denoise_net,"curr_model.pt")            
            print("model saved for loss:",str(loss.item()))
 
        if batch_count==maximum_iteration or loss<=stop_epsilon:
            exit()
               
        loss.backward()
        optimizer.step()