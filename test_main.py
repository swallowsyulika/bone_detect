import cv2
import numpy as np
import os 
import pyopenpose as op
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
from myDataset import MyDataset
from model import Net

torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

transforms_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])
data_list = ['normal', 'bad_leg']

ds = MyDataset(data_list, transforms_)
print(ds[5][0], ds[5][1])

input = ds[5][0].to(device)

net  = Net()
net.to(device)
net.load_state_dict(torch.load("weights/E050.pth"))

output = net(input)
output = output.cpu()
output = torch.argmax(output, dim=1)
if output == 1:
    print('Good')
elif output == 0:
    print('U R so Bad')

