import torch
import torch.nn.functional as F


dim1 = torch.rand(5)
dim2 = torch.nn.functional.pad(dim1, (0,0))
nfp = torch.nn.functional.pad(dim1, (0,6))

t = torch.rand(5, 4)
x = torch.nn.functional.pad(t, 1)
other = torch.rand(5, 8)
# b = t.view(2, 8)
# t1 = t.expand_as(other)
out = F.pad(t, other, "constant", 0)
print(t)
