import torch
from torch import nn
import torch.optim as optim
from XOR import XOR

data_loader = [[(1, 1), (1, 0)], 
               [(1, 0), (0, 1)],  
               [(0, 1), (0, 1)], 
               [(0, 0), (1, 0)]]


epochs = 20000
learning_rate = .1
device = "cpu"

net = XOR()   #new net
net.to(device)

optimizer = optim.SGD(net.parameters(), lr=learning_rate) 
loss_func = nn.MSELoss()
