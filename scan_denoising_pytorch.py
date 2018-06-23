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
import matplotlib.pyplot as plt
import argparse

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Documents Denoising')
parser.add_argument('--mode', type=str, default="train")
opt = parser.parse_args()
print(opt)

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
        input = x
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

#gradient computation
x_grad = np.array([[1, 0, -1],[2, 0,-2],[ 1, 0,-1]])
y_grad = np.array([[1, 2,  1],[0, 0, 0],[-1,-2,-1]])
conv_grad_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
conv_grad_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
conv_grad_x.weight=nn.Parameter(torch.from_numpy(x_grad).float().unsqueeze(0).unsqueeze(0).cuda())
conv_grad_y.weight=nn.Parameter(torch.from_numpy(y_grad).float().unsqueeze(0).unsqueeze(0).cuda())
def gradient(img):
    img_x = conv_grad_x(img)
    img_y = conv_grad_y(img)
    grad = torch.sqrt(torch.pow(img_x,2)+ torch.pow(img_y,2))
    return grad

#compute loss
ave = 1.0/(28*28*50)
def MyLoss(src,dst):
    grad_diff = (gradient(src)-gradient(dst)).abs().sum().mul(ave*0.28)
    abs_diff = (src-dst).abs().sum().mul(ave*0.72)
    return grad_diff,abs_diff


training = (opt.mode == "train")

dataset = DocDataset()
loader = DataLoader(dataset,batch_size=50,shuffle=True)

if training:
    denoise_net = DocCleanNet()
else:
    denoise_net = torch.load("curr_model.pt")

denoise_net.cuda()

#test the sample picture
def test(image_file,model,save_file):
    result = prepare_data.prepare_test(image_file)
    w = result["w"]
    h = result["h"]
    dirty_patches = result["patches"]
    patches = torch.Tensor(result["patches"])
    patches = patches.view(-1,1,28,28)
    patches = patches.cuda()
    patches = Variable(patches)
    predictions = model(patches).view(-1,28,28)
    clean_patches = list(predictions.data.cpu().numpy())
    reconstructed_image, reconstructed_image_dirty = prepare_data.reconstruct_image(w, h, clean_patches, dirty_patches)
    plt.imshow(reconstructed_image,cmap="gray")
    plt.savefig(save_file)
    plt.waitforbuttonpress()
    
if training:
    record = open("loss.txt","w")
    min_loss = 99999
    step = 0.0001
    optimizer = optim.Adam(denoise_net.parameters(),lr=step)
    for epoch in range(1000):
        denoise_net.train()
        for batch_id,[dirty_batch,clean_batch] in enumerate(loader):
            dirty = Variable(dirty_batch.view(-1,1,28,28).cuda())
            clean = Variable(clean_batch.view(-1,1,28,28).cuda())
            optimizer.zero_grad()
            out = denoise_net(dirty)
            grad_loss,abs_loss = MyLoss(out,clean)
            record.write(str(abs_loss.item())+","+str(grad_loss.item())+"\n")
            loss = grad_loss+abs_loss
            if((loss.item()<0.03) and (loss.item()<min_loss)):
                step = step/5
                optimizer = optim.Adam(denoise_net.parameters(),lr=step)
                torch.save(denoise_net,"curr_model.pt")
                min_loss = loss.item()
                print("model saved for loss:",str(min_loss))
                if(min_loss<0.01):
                    record.close()
                    test("./test/1.png",denoise_net,"result.png")                
            if(batch_id%100==0):
                print(abs_loss.data.cpu().numpy())
                print(grad_loss.data.cpu().numpy())            
                print("\n")
            loss.backward()
            optimizer.step()
else:
    test("./test/1.png",denoise_net,"result.png")
