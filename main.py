import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from XOR import *
from train import train
from test import test
from data import *   #net is in data

print(torch.__version__)
print(f'using {device} as device')

#all inputs here specified in data.py
train(data_loader, net, epochs, optimizer, loss_func, device) 

test(data_loader, net, device)








