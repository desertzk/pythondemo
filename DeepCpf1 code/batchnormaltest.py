import torch.nn as nn
import torch
import numpy as np

m = nn.BatchNorm1d(3)
# Without Learnable Parameters
m = nn.BatchNorm1d(3, affine=False)
input = np.array([[1.,5.,7.],[2.,3.,4.],[1,2,3]],dtype=np.float)
input = torch.from_numpy(input).to(torch.float32)

output = m(input)

m = nn.BatchNorm1d(100)
# Without Learnable Parameters
m = nn.BatchNorm1d(100, affine=False)
input = torch.randn(20, 100)
output = m(input)


m = nn.BatchNorm2d(100)
# Without Learnable Parameters
m = nn.BatchNorm2d(100, affine=False)
input = torch.randn(20, 100, 35, 45)
output = m(input)