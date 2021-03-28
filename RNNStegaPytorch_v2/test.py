import torch
from torch import nn
import torch.nn.functional
a=torch.FloatTensor([1,2,3])
b=torch.nn.functional.softmax(a,dim=0)
print(b)