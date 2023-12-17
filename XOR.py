import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

class XOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 5, bias=True)
        self.layer2 = nn.Linear(5, 5, bias=True)
        self.layer3 = nn.Linear(5, 2, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.softMax = nn.Softmax(dim=0) 
        
    def forward(self, x):
        x = self.layer1(x)       #first layer
        x = self.sigmoid(x)      #sigmoid activation
        x = self.layer2(x)       #second layer
        x = self.sigmoid(x)      #sigmoid activation
        x = self.layer3(x)       #third layer
        x = self.softMax(x)      #softmax activation
        return x                 #return output probabilities 