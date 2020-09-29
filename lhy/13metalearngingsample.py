import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import pandas as pd
import random
import seaborn as sbs
from tqdm import tqdm_notebook as tqdm


# https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0

class SineWaveTask:
    def __init__(self):
        self.a = np.random.uniform(0.1, 5.0)
        self.b = np.random.uniform(0, 2 * np.pi)
        self.train_x = None

    def f(self, x):
        return self.a * np.sin(x + self.b)

    def training_set(self, size=10, force_new=False):
        if self.train_x is None and not force_new:
            self.train_x = np.random.uniform(-5, 5, size)
            x = self.train_x
        elif not force_new:
            x = self.train_x
        else:
            x = np.random.uniform(-5, 5, size)
        y = self.f(x)
        return torch.Tensor(x), torch.Tensor(y)

    def test_set(self, size=50):
        x = np.linspace(-5, 5, size)
        y = self.f(x)
        return torch.Tensor(x), torch.Tensor(y)

    def plot(self, *args, **kwargs):
        x, y = self.test_set(size=100)
        return plt.plot(x.numpy(), y.numpy(), *args, **kwargs)


SineWaveTask().plot()
SineWaveTask().plot()
SineWaveTask().plot()
plt.show()

all_x, all_y = [], []

for _ in range(10000):
    curx, cury = SineWaveTask().test_set(size=100)
    all_x.append(curx.numpy())
    all_y.append(cury.numpy())

avg, = plt.plot(all_x[0], np.mean(all_y, axis=0))
rand, = SineWaveTask().plot()
plt.legend([avg, rand], ['Average', 'Random'])
plt.show()
'''
The average is basically 0, which means a neural network trained on a lot of tasks would simply return 0 everywhere! It 
is unclear that this will actually help very much, and yet this is the transfer learning approach in this case...
'''

TRAIN_SIZE = 10000
TEST_SIZE = 1000


class ModifiableModule(nn.Module):
    def params(self):
        return [p for _, p in self.named_params()]

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self):
        subparams = []
        for name, mod in self.named_submodules():
            for subname, param in mod.named_params():
                subparams.append((name + '.' + subname, param))
        return self.named_leaves() + subparams

    def set_param(self, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in self.named_submodules():
                if module_name == name:
                    mod.set_param(rest, param)
                    break
        else:
            setattr(self, name, param)

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = V(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class GradLinear(ModifiableModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.weights = V(ignore.weight.data, requires_grad=True)
        self.bias = V(ignore.bias.data, requires_grad=True)

    def forward(self, x):
        return F.linear(x, self.weights, self.bias)

    def named_leaves(self):
        return [('weights', self.weights), ('bias', self.bias)]


class SineModel(ModifiableModule):
    def __init__(self):
        super().__init__()
        self.hidden1 = GradLinear(1, 40)
        self.hidden2 = GradLinear(40, 40)
        self.out = GradLinear(40, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)

    def named_submodules(self):
        return [('hidden1', self.hidden1), ('hidden2', self.hidden2), ('out', self.out)]


SINE_TRAIN = [SineWaveTask() for _ in range(TRAIN_SIZE)]
SINE_TEST = [SineWaveTask() for _ in range(TEST_SIZE)]

ONE_SIDED_EXAMPLE = None
while ONE_SIDED_EXAMPLE is None:
    cur = SineWaveTask()
    x, _ = cur.training_set()
    x = x.numpy()
    if np.max(x) < 0 or np.min(x) > 0:
        ONE_SIDED_EXAMPLE = cur

SINE_TRANSFER = SineModel()


def sine_fit1(net, wave, optim=None, get_test_loss=False, create_graph=False, force_new=False):
    net.train()
    if optim is not None:
        optim.zero_grad()
    x, y = wave.training_set(force_new=force_new)
    loss = F.mse_loss(net(V(x[:, None])), V(y).unsqueeze(1))
    loss.backward(create_graph=create_graph, retain_graph=True)
    if optim is not None:
        optim.step()
    if get_test_loss:
        net.eval()
        x, y = wave.test_set()
        loss_test = F.mse_loss(net(V(x[:, None])), V(y))
        return loss.data.cpu().numpy()[0], loss_test.data.cpu().numpy()[0]
    return loss.data.cpu().numpy()  # [0]


def fit_transfer(epochs=1):
    optim = torch.optim.Adam(SINE_TRANSFER.params())

    for _ in range(epochs):
        for t in random.sample(SINE_TRAIN, len(SINE_TRAIN)):
            sine_fit1(SINE_TRANSFER, t, optim)


fit_transfer()


def copy_sine_model(model):
    m = SineModel()
    m.copy(model)
    return m


def eval_sine_test(model, test, fits=(0, 1), lr=0.01):
    xtest, ytest = test.test_set()
    xtrain, ytrain = test.training_set()

    model = copy_sine_model(model)
    # Not sure if this should be Adam or SGD.
    optim = torch.optim.SGD(model.params(), lr)

    def get_loss(res):
        return F.mse_loss(res, V(ytest[:, None])).cpu().data.numpy()  # [0]

    fit_res = []
    if 0 in fits:
        results = model(V(xtest[:, None]))
        fit_res.append((0, results, get_loss(results)))
    for i in range(np.max(fits)):
        sine_fit1(model, test, optim)
        if i + 1 in fits:
            results = model(V(xtest[:, None]))
            fit_res.append(
                (
                    i + 1,
                    results,
                    get_loss(results)
                )
            )

    return fit_res


def plot_sine_test(model, test, fits=(0, 1), lr=0.01):
    xtest, ytest = test.test_set()
    xtrain, ytrain = test.training_set()

    fit_res = eval_sine_test(model, test, fits, lr)

    train, = plt.plot(xtrain.numpy(), ytrain.numpy(), '^')
    ground_truth, = plt.plot(xtest.numpy(), ytest.numpy())
    plots = [train, ground_truth]
    legend = ['Training Points', 'True Function']
    for n, res, loss in fit_res:
        cur, = plt.plot(xtest.numpy(), res.cpu().data.numpy()[:, 0], '--')
        plots.append(cur)
        legend.append(f'After {n} Steps')
    plt.legend(plots, legend)
    plt.show()


plot_sine_test(SINE_TRANSFER, SINE_TEST[0], fits=[0, 1, 10], lr=0.02)


def plot_sine_learning(models, fits=(0, 1), lr=0.01, marker='s', linestyle='--'):
    data = {'model': [], 'fits': [], 'loss': [], 'set': []}
    for name, models in models:
        if not isinstance(models, list):
            models = [models]
        for n_model, model in enumerate(models):
            for n_test, test in enumerate(SINE_TEST):
                n_test = n_model * len(SINE_TEST) + n_test
                fit_res = eval_sine_test(model, test, fits, lr)
                for n, _, loss in fit_res:
                    data['model'].append(name)
                    data['fits'].append(n)
                    data['loss'].append(loss)
                    data['set'].append(n_test)

    ax = sbs.tsplot(
        pd.DataFrame(data), condition='model', value='loss',
        time='fits', unit='set', marker=marker, linestyle=linestyle)


plot_sine_learning(
    [('Transfer', SINE_TRANSFER), ('Random', SineModel())],
    list(range(100)),
    marker='',
    linestyle='-'
)


'''
MAML
We now come to MAML, the first of the two algorithms we will look at today.

As mentioned before, we are trying to find a set of weights such that running gradient descent on similar tasks makes progress 
as quickly as possible. MAML takes this extremely literally by running one iteration of gradient descent and then updating the 
initial weights based on how much progress that one iteration made towards the true task. More concretely it:

Creates a copy of the initialization weights
Runs an iteration of gradient descent for a random task on the copy
Backpropagates the loss on a test set through the iteration of gradient descent and back to the initial weights, so that 
we can update the initial weights in a direction in which they would have been easier to update.
We thus need to take a gradient of a gradient, aka a second degree derivative in this process. Fortunately this is something 
that PyTorch supports now, unfortunately PyTorch makes it a bit awkward to update the parameters of a model in a way that 
we can still run gradient descent through them (we already saw this is "Learning to Learn by Gradient Descent by Gradient Descent"),
 which explains the weird way in which the model is written.

Because we are going to use second derivatives, we need to make sure that the computational graph that allowed us to compute 
the original gradients stays around, which is why we pass create_graph=True to .backward().

The code below also implements first order MAML, which we explain later:
'''