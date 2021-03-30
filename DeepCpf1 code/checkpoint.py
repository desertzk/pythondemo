import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

torch.manual_seed(2809)
np.random.seed(2809)


# Create simple model
class MyModel(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.drop1 = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.bn1(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def train(model, optimizer):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss.data[0]


def print_weights(model):
    print(model.fc1.weight.data)
    print(model.fc2.weight.data)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


checkpoint_path = 'checkpoint.pth.tar'
batch_size = 10
in_features = 5
hidden = 10
out_features = 3

model = MyModel(in_features, hidden, out_features)
model.apply(weight_init)
print_weights(model)

optimizer = optim.Adam(model.parameters(),
                       lr=5e-4,
                       betas=(0.9, 0.999),
                       eps=1e-08,
                       weight_decay=1e-4)

criterion = nn.NLLLoss()

x = Variable(torch.randn(batch_size, in_features))
y = Variable(torch.LongTensor(batch_size).random_(out_features))

# Train for one epoch
losses = []
for epoch in range(10):
    loss_val = train(model, optimizer)
    losses.append(loss_val)
print_weights(model)

losses_arr = np.array(losses)
plt.plot(losses_arr)

save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}, checkpoint_path)

# Resume training
model_resume = MyModel(in_features, hidden, out_features)
optimizer_resume = optim.Adam(model_resume.parameters(),
                              lr=5e-4,
                              betas=(0.9, 0.999),
                              eps=1e-08,
                              weight_decay=1e-4)

print("=> loading checkpoint '{}'".format(checkpoint_path))
checkpoint = torch.load(checkpoint_path)
start_epoch = checkpoint['epoch']
model_resume.load_state_dict(checkpoint['state_dict'])
optimizer_resume.load_state_dict(checkpoint['optimizer'])
print("=> loaded checkpoint '{}' (epoch {})"
      .format(checkpoint_path, checkpoint['epoch']))

print_weights(model)
print_weights(model_resume)

# Train models for 10 epochs
losses_resume = losses
for epoch in range(10, 20):
    loss_val = train(model, optimizer)
    losses.append(loss_val)

    loss_val_resume = train(model_resume, optimizer_resume)
    losses_resume.append(loss_val_resume)

losses_arr = np.array(losses)
losses_resume_arr = np.array(losses_resume)

plt.plot(losses_arr)
plt.plot(losses_resume_arr)

if not np.allclose(losses_arr, losses_resume_arr):
    print('Losses are different!')