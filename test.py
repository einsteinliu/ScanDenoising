import os
import torch
import torch.nn.functional as Func
import cv2
import configparser
import argparse
import numpy as np
import matplotlib.pyplot as plt
from prepare_data import adaptImageToNet
from os.path import join

# Training settings
cmdparser = argparse.ArgumentParser(description='PyTorch Corner')
cmdparser.add_argument('--config', type=str, default="config.ini")
opt = cmdparser.parse_args()
print(opt)
parser = configparser.ConfigParser()
parser.read(opt.config)

folder = parser['test']['folder']
model_file = parser['test']['model_file']
output_folder = parser['test']['output_folder']

if not os.path.exists(model_file):
    print(model_file+' doesn''t exist')
    exit()

if not os.path.exists(folder):
    print(folder+' doesn''t exist')
    exit()

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

files = os.listdir(folder)
images = [file for file in files if '.png' in file]

if len(images)==0:
    print('No images in folder '+folder)
    exit()

model = torch.load(model_file)

for image_file in images:
    print('predict '+image_file)
    origin = cv2.imread(join(folder,image_file),0)
    image = adaptImageToNet(origin_image=origin,div=8)
    patch = torch.Tensor(image).view(-1,1,image.shape[0],image.shape[1])    
    patch = patch.cuda()
    clean_image = model(patch).view(-1,image.shape[0],image.shape[1])
    clean_image = clean_image.data.cpu().numpy()[0,:,:]
    #cv2.imshow('clean',clean_image)
    #cv2.waitKey(-1)
    clean_image = clean_image*255.0
    cv2.imwrite(join(output_folder,image_file),clean_image)   