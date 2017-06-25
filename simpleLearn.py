import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from simple_dataset import SimpleDataset
from simpleOptimizer import SimpleOptimizer

class Simple(nn.Module):

        def __init__(self, indim,outdim):
                #For the time being, let's work with 2 activations: relu and identity
                super(Simple, self).__init__()
                self.applyW=nn.Linear(indim,outdim)
        def forward(self,x):
                out=self.applyW(x)
                return out





myDb=SimpleDataset()
dataLoader = DataLoader(myDb, batch_size=3,
                        shuffle=True, num_workers=0)

simple=Simple(2,1)
import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.SGD(simple.parameters(), lr=0.001)



NUMBER_OF_EPOCHS=20

for _ in range(NUMBER_OF_EPOCHS):

        for (example_num,data) in enumerate(dataLoader):
                optimizer.zero_grad()
                inputs=Variable(data['example'].float())
                targets=Variable(data['target'].float())
                outputs=simple(inputs)
                loss = criterion(outputs,targets)
                loss.backward()
                optimizer.step()
                params=[p for p in simple.parameters()]


