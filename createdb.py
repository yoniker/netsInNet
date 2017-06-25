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


FILENAME='v1pv2VERSION2.csv'
ATTRIBUTES=['v1','v2']
NUM_EXAMPLES=200000


f = open(FILENAME, 'w')
for attribute in ATTRIBUTES:
	f.write(attribute+',')
f.write('result\n')

for _ in range(NUM_EXAMPLES):
	v1=np.random.randint(-1000,1000)/100.0
	v2=np.random.randint(-1000,1000)/100.0
	result=v1+v2
	f.write("{},{},{}\n".format(v1,v2,result))



f.close()