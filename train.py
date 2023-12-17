import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import random

def train(data_loader, net, epochs, optimizer, loss_func, device):
    net.train()
    for epoch in range(epochs):
        random.shuffle(data_loader)
        totalLoss = 0
        for input, target in data_loader:
            input = torch.tensor(input, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            
            output = net(input)
            
            loss = loss_func(output, target)
            totalLoss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 1000 == 0:
            print("Epoch: {: >8} Loss: {}".format(epoch+1, totalLoss/len(data_loader)))
            
            
        
        
        
        