import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

# 生成  𝑎∗sin(𝑥+𝑏)  的資料點，其中 𝑎,𝑏 的範圍分別預設為 [0.1,5],[0,2𝜋] ，
# 每一個  𝑎∗sin(𝑥+𝑏) 的函數有10個資料點當作訓練資料。測試時則可用較密集的資料點直接由畫圖來看generalize的好壞。
# 这里其实就是sample数据
device = 'cpu'
def meta_task_data(seed = 0, a_range=[0.1, 5], b_range = [0, 2*np.pi], task_num = 100,
                   n_sample = 10, sample_range = [-5, 5], plot = False):
    np.random.seed = seed
    a_s = np.random.uniform(low = a_range[0], high = a_range[1], size = task_num)  #0.1   5从均匀分布中取值 Uniform distribution
    b_s = np.random.uniform(low = b_range[0], high = b_range[1], size = task_num)
    total_x = []
    total_y = []
    label = []
    for t in range(task_num):
        x = np.random.uniform(low = sample_range[0], high = sample_range[1], size = n_sample) # -5 5的均匀分布中取值 取10个值
        total_x.append(x)
        total_y.append( a_s[t]*np.sin(x+b_s[t]) )
        label.append('{:.3}*sin(x+{:.3})'.format(a_s[t], b_s[t]))
    if plot:
        plot_x = [np.linspace(-5, 5, 1000)]  #返回1000个 从-5 到5之间的均匀样本
        plot_y = []
        for t in range(task_num):
            plot_y.append( a_s[t]*np.sin(plot_x+b_s[t]) )
        return total_x, total_y, plot_x, plot_y, label
    else:
        return total_x, total_y, label


'''
以下我們將老師MAML投影片第27頁的 𝜙 稱作meta weight φ是初始化参数， 𝜃 則稱為sub weight。

老師投影片： http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Meta1%20(v6).pdf

為了讓sub weight的gradient能夠傳到meta weight (因為sub weight的初始化是從meta weight來的，所以想當然我們用sub weight算出來的loss對
meta weight也應該是可以算gradient才對)，這邊我們需要重新定義一些pytorch內的layer的運算。

實際上MetaLinear這個class做的事情跟torch.nn.Linear完全是一樣的，唯一的差別在於這邊的每一個tensor都沒有被變成torch.nn.Parameter。
這麼做的原因是因為等一下我們從meta weight那裏複製(init weight輸入meta weight後weight與bias使用.clone)的時候，tensor的clone的操作
是可以傳遞gradient的，以方便我們用gradient更新meta weight。這個寫法的代價是我們就沒辦法使用torch.optim更新sub weight了，因為參數都只用tensor紀錄。
也因此我們接下來需要自己寫gradient update的函數(只用SGD的話是簡單的)。
'''

class MetaLinear(nn.Module):
    '''
    拿一样的初始化参数去各个task上训练
，最终学出来的model是不一样的
    '''
    def __init__(self, init_layer = None):
        super(MetaLinear, self).__init__()
        if type(init_layer) != type(None):
            self.weight = init_layer.weight.clone()
            self.bias = init_layer.bias.clone()
    def zero_grad(self):
        self.weight.grad  = torch.zeros_like(self.weight)
        self.bias.grad  = torch.zeros_like(self.bias)
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


'''
這裡的forward和一般的model是一樣的，唯一的差別在於我們需要多寫一下__init__函數讓他比起一般的pytorch model多一個可以從meta weight(𝜙)複製
的功能(這邊因為我把model的架構寫死了所以可能看起來會有點多餘，讀者可以自己把net()改成可以自己調整架構的樣子，然後思考一下如何生成一個跟
meta weight一樣形狀的sub weight)

update函數就如同前一段提到的，我們需要自己先手動用SGD更新一次sub weight，接著再使用下一步的gradient(第二步)來更新
meta weight。zero_grad函數在此處沒有用到，因為實際上我們計算第二步的gradient時會需要第一步的grad，這也是為什麼我們第一次backward的時候
需要create_graph=True (建立計算圖以計算二階的gradient)
'''


class net(nn.Module):
    def __init__(self, init_weight=None):
        super(net, self).__init__()
        if type(init_weight) != type(None):
            for name, module in init_weight.named_modules():
                if name != '':
                    setattr(self, name, MetaLinear(module))
        else:
            self.hidden1 = nn.Linear(1, 40)   #input 1 output 40
            self.hidden2 = nn.Linear(40, 40)   #input 40 output 40
            self.out = nn.Linear(40, 1)      #input 40 output 1

    def zero_grad(self):
        layers = self.__dict__['_modules']
        for layer in layers.keys():
            layers[layer].zero_grad()
    # 这里的parent是指Meta_learning_model  这里用Meta_learning_model的grad进行更新
    def update(self, parent, lr=1):
        layers = self.__dict__['_modules']
        parent_layers = parent.__dict__['_modules']
        for param in layers.keys():
            layers[param].weight = layers[param].weight - lr * parent_layers[param].weight.grad
            layers[param].bias = layers[param].bias - lr * parent_layers[param].bias.grad
        # gradient will flow back due to clone backward

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)

'''
前面的class中我們已經都將複製meta weight到sub weight，以及sub weight的更新，gradient的傳遞都搞定了，meta weight自己本身的參數就
可以按照一般pytorch model的模式，使用torch.optim來更新了。

gen_model函數做的事情其實就是產生N個sub weight，並且使用前面我們寫好的複製meta weight的功能。

注意到複製weight其實是整個code的關鍵，因為我們需要將sub weight計算的第二步gradient正確的傳回meta weight。讀者從meta weight與
sub weight更新參數作法的差別(手動更新/用torch.nn.Parameter與torch.optim)可以再思考一下兩者的差別。
'''

class Meta_learning_model():
    def __init__(self, init_weight = None):
        super(Meta_learning_model, self).__init__()
        self.model = net().to(device)
        if type(init_weight) != type(None):
            self.model.load_state_dict(init_weight)
        self.grad_buffer = 0
    def gen_models(self, num, check = True):
        models = [net(init_weight=self.model).to(device) for i in range(num)]  #用一开始设置好的model架构Linear(1,40)  Linear(40,40) Linear(40,1)生成
        return models
    def clear_buffer(self):
        print("Before grad", self.grad_buffer)
        self.grad_buffer = 0


# 接下來就是生成訓練/測試資料，建立meta weightmeta weight的模型以及用來比較的model pretraining的模型
# batch size 10 代表每一轮执行10个任务
bsz = 10
# 总共生成 50000*10 个任务 task
train_x, train_y, train_label = meta_task_data(task_num=50000*10)
train_x = torch.Tensor(train_x).unsqueeze(-1) # add one dim 从（50000,10) 变成 （50000,10,1)
train_y = torch.Tensor(train_y).unsqueeze(-1) # y = 𝑎∗sin(𝑥+𝑏)   从（50000,10) 变成 （50000,10,1)
# Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作。
# DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作), num_workers(加载数据的时候使用几个子进程)
train_dataset = data.TensorDataset(train_x, train_y)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=bsz, shuffle=False)

test_x, test_y, plot_x, plot_y, test_label = meta_task_data(task_num=1, n_sample = 10, plot=True)
test_x = torch.Tensor(test_x).unsqueeze(-1) # 1,10,1 add one dim
test_y = torch.Tensor(test_y).unsqueeze(-1) # add one dim
plot_x = torch.Tensor(plot_x).unsqueeze(-1) # add one dim
test_dataset = data.TensorDataset(test_x, test_y)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=bsz, shuffle=False)


meta_model = Meta_learning_model()

meta_optimizer = torch.optim.Adam(meta_model.model.parameters(), lr = 1e-3)


pretrain = net()
pretrain.to(device)
pretrain.train()
pretrain_optim = torch.optim.Adam(pretrain.parameters(), lr = 1e-3)

'''
進行訓練，注意一開始我們要先生成一群sub weight(code裡面的sub models)，然後將一個batch的不同的sin函數的10筆資料點拿來訓練sub weight。
注意這邊sub weight計算第一步gradient與第二步gradient時使用各五筆不重複的資料點(因此使用[:5]與[5:]來取)。但在訓練model pretraining的
對照組時則沒有這個問題(所以pretraining的model是可以確實的走兩步gradient的)

每一個sub weight計算完loss後相加(內層的for迴圈)後就可以使用optimizer來更新meta weight，再次提醒一下sub weight計算第一次loss的時候
backward是需要create_graph=True的，這樣計算第二步gradient的時候才會真的計算到二階的項。讀者可以在這個地方思考一下如何將這段程式碼改成MAML的一階做法。
'''
epoch = 1
for e in range(epoch):
    meta_model.model.train()
    for x, y in tqdm(train_loader):  #这里就是一轮 一轮的数据量就是batch size 就是 10
        x = x.to(device)    #这里的x是 【10,10,1】   第一个是batch size 10   第二个 10个数据点吧
        y = y.to(device)    #这里的y是 【10,10,1】
        sub_models = meta_model.gen_models(bsz)  #一開始我們要先生成一群（这里是10个）sub weight(code裡面的sub models)

        meta_l = 0
        for model_num in range(len(sub_models)):
            sample = torch.randint(0, 10, size=(10,), dtype=torch.long)

            # pretraining
            pretrain_optim.zero_grad()
            y_tilde = pretrain(x[model_num][sample[:5], :]) #取出抽样中的前五个下标对应的 第model_num个sub model input x里的值  这里算出来的y_tilde是前向传播的值
            little_l = F.mse_loss(y_tilde, y[model_num][sample[:5], :]) #loss
            little_l.backward()
            pretrain_optim.step()  #优化一步 只有用了optimizer.step()，模型才会更新
            pretrain_optim.zero_grad()
            y_tilde = pretrain(x[model_num][sample[5:], :]) #取出抽样中的后五个下标对应的 第model_num个sub model input x里的值
            little_l = F.mse_loss(y_tilde, y[model_num][sample[5:], :])
            little_l.backward()  #反向传播
            pretrain_optim.step()  #优化第二步 这里做到二阶微分 用的是不同数据

            # meta learning
            # sub weight計算第一步gradient與第二步gradient時使用各五筆不重複的資料點(因此使用[:5]與[5:]來取)
            y_tilde = sub_models[model_num](x[model_num][sample[:5], :])   #这里会调用前项传播
            little_l = F.mse_loss(y_tilde, y[model_num][sample[:5], :]) # loss
            # 計算第一次gradient並保留計算圖以接著計算更高階的gradient
            little_l.backward(create_graph=True)
            sub_models[model_num].update(lr=1e-2, parent=meta_model.model)  #自己做更新参数 这里为什么要自己写update而不是用pytorch的step？
            # 先清空optimizer中計算的gradient值(避免累加)
            meta_optimizer.zero_grad()

            # 計算第二次(二階)的gradient，二階的原因來自第一次update時有計算過一次gradient了
            y_tilde = sub_models[model_num](x[model_num][sample[5:], :])
            meta_l = meta_l + F.mse_loss(y_tilde, y[model_num][sample[5:], :]) #这里是meta learning每一个task累加？ 相当于 task的loss

        meta_l = meta_l / bsz
        meta_l.backward() #backward代表计算gradient
        meta_optimizer.step()
        meta_optimizer.zero_grad()

# 測試我們訓練好的meta weight
test_model = copy.deepcopy(meta_model.model)
test_model.train()
test_optim = torch.optim.SGD(test_model.parameters(), lr = 1e-3)

# 先畫出待測試的sin函數，以及用圓點點出測試時給meta weight訓練的十筆資料點
fig = plt.figure(figsize = [9.6,7.2])
ax = plt.subplot(111)
plot_x1 = plot_x.squeeze().numpy()
ax.scatter(test_x.numpy().squeeze(), test_y.numpy().squeeze())
ax.plot(plot_x1, plot_y[0].squeeze())

# 分別利用十筆資料點更新meta weight以及pretrained model一個step
test_model.train()
pretrain.train()

for epoch in range(1):
    for x, y in test_loader:
        y_tilde = test_model(x[0])
        little_l = F.mse_loss(y_tilde, y[0])
        test_optim.zero_grad()
        little_l.backward()
        test_optim.step()
        print("(meta)))Loss: ", little_l.item())

for epoch in range(1):
    for x, y in test_loader:
        y_tilde = pretrain(x[0])
        little_l = F.mse_loss(y_tilde, y[0])
        pretrain_optim.zero_grad()
        little_l.backward()
        pretrain_optim.step()
        print("(pretrain)Loss: ", little_l.item())

# 將更新後的模型所代表的函數繪製出來，與真實的sin函數比較
test_model.eval()
pretrain.eval()

plot_y_tilde = test_model(plot_x[0]).squeeze().detach().numpy()
plot_x2 = plot_x.squeeze().numpy()
ax.plot(plot_x2, plot_y_tilde, label = 'tune(disjoint)')
ax.legend()
# fig.show()

#
plot_y_tilde = pretrain(plot_x[0]).squeeze().detach().numpy()
plot_x2 = plot_x.squeeze().numpy()
ax.plot(plot_x2, plot_y_tilde, label = 'pretrain')
ax.legend()

# 執行底下的cell以顯示圖形，並重複執行更新meta weight與pretrained model的cell來比較多更新幾步後是否真的能看出meta learning比model pretraining有效
plt.show()





