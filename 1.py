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
import torch.nn.functional as F
from simple_dataset import SimpleDataset


class MyFCLayer(nn.Module):
        def __init__(self, indim,outdim):
                #For the time being, let's work with 2 activations: relu and identity
                super(MyFCLayer, self).__init__()
                #self.activations=activations_list
                #self.k=len(activations_list)
                #I want to have K different weights.
                self.k=2 #TODO apply any kind of activation later.
                self.applyW1= nn.Linear(indim, outdim)
                self.applyW2= nn.Linear(indim, outdim)
                self.lambdas_matrix=Parameter(torch.ones(outdim,self.k))
                self.softmax = nn.Softmax()
                for parameter in self.applyW1.parameters():
                        parameter.lambda_index=0
                for parameter in self.applyW2.parameters():
                        parameter.lambda_index=1 #TODO find a more elegant way to do that!


        def forward(self,X):
                Act1=self.applyW1(X)
                Act1=F.relu((Act1))
                Act2=self.applyW2(X)
                #identity so do nothing with Act2
                Act_weights=self.softmax(self.lambdas_matrix)
                self.current_lambdas=Act_weights
                Act=torch.cat([Act1,Act2],1)
                after_weighing=Act_weights.expand_as(Act)*Act
                output=after_weighing.sum(1)
                return output

        #a method which will change the gradient of all the weights- divide the relevant weights by their lambdas
        def change_gradients(self):
                for parameter in self.parameters():
                        if hasattr(parameter, 'lambda_index'):
                                parameter.grad.data=torch.div(parameter.grad.data,self.current_lambdas[:,parameter.lambda_index].data.expand_as(parameter.grad.data))


from simple_dataset import SimpleDataset
myDb=SimpleDataset()
dataLoader = DataLoader(myDb, batch_size=3,
                        shuffle=True, num_workers=0)


new_model=MyFCLayer(2,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(new_model.parameters(), lr=0.001)


def train(number_of_epochs):

        for _ in range(number_of_epochs):

                for (example_num,data) in enumerate(dataLoader):
                        optimizer.zero_grad()
                        inputs=Variable(data['example'].float())
                        targets=Variable(data['target'].float())
                        outputs=new_model(inputs)
                        loss = criterion(outputs,targets)
                        loss.backward()
                        #Change the gradient so that it will be grad/lambda - each subsystem will train at full pace,regardless of its current contribution to the output.
                        new_model.change_gradients()
                        optimizer.step()

train(1)




# for (example_num,data) in enumerate(dataLoader):
#         optimizer.zero_grad()
#         inputs=Variable(data['example'].float())
#         targets=Variable(data['target'].float())
#         outputs=new_model(inputs)
#         loss = criterion(outputs,targets)
#         loss.backward()
#         break
#         #Change the gradient so that it will be grad/lambda - each subsystem will train at full pace,regardless of its current contribution to the output.
#         #new_model.lambdas_matrix.grad.data=(new_model.lambdas_matrix.grad/new_model.current_lambdas).data
#         optimizer.step()


