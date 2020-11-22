import numpy as np
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from numpy import zeros
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.distributions import Categorical
import time
import math
from matplotlib import pyplot as plt
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
# cudnn.benchmark = True


def PREPROCESS_ONE_HOT(train_data):
    data_n = len(train_data)
    SEQ = zeros((data_n, 34, 4), dtype=int)
    # CA = zeros((data_n, 1), dtype=int)

    for l in range(0, data_n):

        seq = train_data[l]
        for i in range(34):
            if seq[i] in "Aa":
                SEQ[l, i, 0] = 1
            elif seq[i] in "Cc":
                SEQ[l, i, 1] = 1
            elif seq[i] in "Gg":
                SEQ[l, i, 2] = 1
            elif seq[i] in "Tt":
                SEQ[l, i, 3] = 1
        # CA[l - 1, 0] = int(data[2])

    return SEQ

def data_load():
    train_data = pd.read_excel('data/41587_2018_BFnbt4061_MOESM39_ESM.xlsx', sheet_name=0)
    test_data = pd.read_excel('data/41587_2018_BFnbt4061_MOESM39_ESM.xlsx', sheet_name=1)
    use_data = train_data[0:14999]
    new_header = test_data.iloc[0]
    test_data = test_data[1:]
    test_data.index = np.arange(0, len(test_data))
    test_data.columns = new_header
    bp34_col = use_data["34 bp synthetic target and target context sequence(4 bp + PAM + 23 bp protospacer + 3 bp)"]
    indel_f = use_data["Indel freqeuncy(Background substracted, %)"]
    SEQ = PREPROCESS_ONE_HOT(bp34_col)

    test_bp34 = test_data["34 bp synthetic target and target context sequence(4 bp + PAM + 23 bp protospacer + 3 bp)"]
    test_indel_f = test_data["Indel freqeuncy(Background substracted, %)"]
    test_SEQ = PREPROCESS_ONE_HOT(test_bp34)
    return SEQ,indel_f,test_SEQ,test_indel_f

class RNADataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x.astype("float32")
        # label is required to be a LongTensor
        self.y = y.astype("float32")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        Y = self.y[index]
        return X, Y

train_x, train_y, test_x, test_y = data_load()
# 维度互换
train_x_for_torch = np.transpose(train_x,(0,2,1))
test_x__for_torch = np.transpose(test_x,(0,2,1))
all_train_set = RNADataset(train_x_for_torch,train_y)
test_set = RNADataset(test_x__for_torch,test_y)
batch_size = len(all_train_set)


train_loader = DataLoader(all_train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

class Regression(nn.Module):
    #(None,34,4)
    def __init__(self,params):
        super(Regression, self).__init__()
        # 这里有1200 4个小数
        conv1d_out_channels = params.get("conv1d_out_channels").get("action")
        conv1d_kernel_size = params.get("conv1d_kernel_size").get("action")
        linear1200_80_out_features = params.get("linear1200_80_out_features").get("action")
        linear80_40_out_features = params.get("linear80_40_out_features").get("action")
        linear40_40_out_features = params.get("linear40_40_out_features").get("action")
        self.need_pool = params.get("need_pool").get("action")
        print("model conv1d_out_channels:",conv1d_out_channels," conv1d_kernel_size:",conv1d_kernel_size," linear1200_80_out_features:",linear1200_80_out_features
              ,"linear80_40_out_features:",linear80_40_out_features," linear40_40_out_features:",linear40_40_out_features," need_pool:",self.need_pool)
        self.conv1d = nn.Conv1d(4, conv1d_out_channels, conv1d_kernel_size, 1) # 进去4通道出来80通道 (30,80)
        self.relu = nn.ReLU()
        self.avg1d = nn.AvgPool1d(2) # size of window 2  (15,80)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)
        # conv1d_kernel_size这里要注意这个conv1d_kernel_size要是奇数不然会有小数点
        if self.need_pool is 1:
            input = int((34-conv1d_kernel_size+1)/2)
        else:
            input = (34 - conv1d_kernel_size + 1)
        self.linear1200_80 = nn.Linear(conv1d_out_channels * input, linear1200_80_out_features)
        self.linear80_40 = nn.Linear(linear1200_80_out_features, linear80_40_out_features)  #(None, 40)
        self.linear40_40 = nn.Linear(linear80_40_out_features, linear40_40_out_features)  # (None, 40)
        self.linear40_1 = nn.Linear(linear40_40_out_features, 1)  # (None, 40)



    def forward(self, x):
        outconv1d = self.conv1d(x) # 进去4通道出来80通道 (30,80)
        outact = self.relu(outconv1d)
        # Seq_deepCpf1_C1 = Convolution1D(80, 5)(Seq_deepCpf1_Input_SEQ)
        if self.need_pool is 1:
            outavg1d = self.avg1d(outact)  # size of window 2  (15,80)
        else:
            outavg1d = outact
        # Seq_deepCpf1_P1 = AveragePooling1D(2)(Seq_deepCpf1_C1)
        out_flatten = self.flatten(outavg1d)
        # Seq_deepCpf1_F = Flatten()(Seq_deepCpf1_P1)

        out_dropout = self.dropout(out_flatten)
        # Seq_deepCpf1_DO1 = Dropout(0.3)(Seq_deepCpf1_F)
        out_linear1200_80 = self.linear1200_80(out_dropout)
        out_act_linear1200_80 = self.relu(out_linear1200_80)
        # Seq_deepCpf1_D1 = Dense(80, activation='relu')(Seq_deepCpf1_DO1)
        out_dropout1200_80 = self.dropout(out_act_linear1200_80)
        # Seq_deepCpf1_DO2 = Dropout(0.3)(Seq_deepCpf1_D1)
        out_linear80_40 = self.linear80_40(out_dropout1200_80)
        out_act80_40 = self.relu(out_linear80_40)
        # Seq_deepCpf1_D2 = Dense(40, activation='relu')(Seq_deepCpf1_DO2)
        out_dropout80_40 = self.dropout(out_act80_40)
        # Seq_deepCpf1_DO3 = Dropout(0.3)(Seq_deepCpf1_D2)
        out_linear40_40 = self.linear40_40(out_dropout80_40)
        out_act40_40 = self.relu(out_linear40_40)
        # Seq_deepCpf1_D3 = Dense(40, activation='relu')(Seq_deepCpf1_DO3)
        out_dropout40_40 = self.dropout(out_act40_40)
        # Seq_deepCpf1_DO4 = Dropout(0.3)(Seq_deepCpf1_D3)
        out = self.linear40_1(out_dropout40_40)
        # Seq_deepCpf1_Output = Dense(1, activation='linear')(Seq_deepCpf1_DO4)
        result = out.squeeze(1)
        return result
        # print(outconv1d.shape)
        # print(outact.shape)
        # print(outavg1d.shape)
        # print(out_flatten.shape)
        # print(out_dropout.shape)
        # print(out_linear1200_80.shape)
        # print(out_act_linear1200_80.shape)
        # print(out_dropout1200_80.shape)
        # print(out_linear80_40.shape)
        # print(out_act80_40.shape)
        # print(out_dropout80_40.shape)
        # print(out_linear40_40.shape)
        # print(out_act40_40.shape)
        # print(out_dropout40_40.shape)
        # print(out.shape)



# 这里有个问题是state怎么搞
# 我的想法是batch里每个输入的atcg碱基作为state
'''

'''
class PolicyGradientNetwork(nn.Module):

    def __init__(self,state,architecture_map=None):
        super().__init__()
        self.architecture_map = architecture_map
        self.len_conv1d_out_channels = len(architecture_map.get("conv1d_out_channels"))
        self.len_conv1d_kernel_size = len(architecture_map.get("conv1d_kernel_size"))
        self.len_linear1200_80_out_features = len(architecture_map.get("linear1200_80_out_features"))
        self.len_linear80_40_out_features = len(architecture_map.get("linear80_40_out_features"))
        self.len_linear40_40_out_features = len(architecture_map.get("linear40_40_out_features"))
        self.len_need_pool = len(architecture_map.get("need_pool"))

        self.flatten = nn.Flatten()
        self.num_param = 4
        state_size = 1
        for i in state:
            state_size = state_size * i
        self.fc1 = nn.Linear(state_size, 64)   # 4*34  64
        self.fc2 = nn.Linear(64, 16)




        self.fc_conv1d_out_channels = nn.Linear(16, self.len_conv1d_out_channels)
        self.fc_conv1d_kernel_size = nn.Linear(16, self.len_conv1d_kernel_size)
        self.fc_linear1200_80_out_features = nn.Linear(16, self.len_linear1200_80_out_features)
        self.fc_linear80_40_out_features = nn.Linear(16, self.len_linear80_40_out_features)
        self.fc_linear40_40_out_features = nn.Linear(16, self.len_linear40_40_out_features)
        self.fc_need_pool = nn.Linear(16, self.len_need_pool)

    def forward(self, state):
        flt = self.flatten(state)
        action_prob_list = []
        hid1 = torch.tanh(self.fc1(flt))
        hid2 = torch.tanh(self.fc2(hid1))

        action_prob_map = {}

        for k, v in self.architecture_map.items():
            linear = nn.Linear(16, len(v))
            result = linear(hid2)
            result_softmax = F.softmax(result, dim=-1)
            action_prob = torch.sum(result_softmax, dim=0)
            action_prob_map[k] = action_prob

        return action_prob_map



# controller
class PolicyGradientAgent():

    def __init__(self):

        self.conv1d_out_channels_list = [80,90,100,110]#[i for i in range(100) if i>9 and i%5==0]
        self.conv1d_kernel_size_list = [7,3,5,1]#[i for i in range(18) if i>0 and i%2==1]
        self.linear1200_80_out_features_list = [70,40,80,60]#[i for i in range(200) if i>9 and i%10==0]
        self.linear80_40_out_features_list = [40,20,35,16]#[i for i in range(150) if i>9 and i%10==0]

        self.linear40_40_out_features_list = [10,20,40,80]#[i for i in range(100) if i>2 and i%10==0]
        self.need_pool = [0,1]
        tp_size = train_x_for_torch[0].shape
        self.architecture_map ={
            "conv1d_out_channels":self.conv1d_out_channels_list,
            "conv1d_kernel_size":self.conv1d_kernel_size_list,
            "linear1200_80_out_features":self.linear1200_80_out_features_list,
            "linear80_40_out_features":self.linear80_40_out_features_list,
            "linear40_40_out_features":self.linear40_40_out_features_list,
            "need_pool":self.need_pool,
        }

        self.network = PolicyGradientNetwork(tp_size,self.architecture_map)#.to(device)
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)


    def learn(self, rewards,log_prob):
        # 损失函数要是一个式子
        loss = -torch.mean(log_prob)*rewards

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob_map = self.network(state) #torch.cuda.FloatTensor(
        # action_prob = torch.sum(item_action_prob,dim=0)   # To sum over all rows (i.e. for each column)  size = [1, ncol]
        regression_params = {}
        total_log_prob = 0
        for k,item_action_prob in action_prob_map.items():
            action_dist = Categorical(item_action_prob)
            action_index = action_dist.sample() #.unsqueeze(1)  这里就是根据概率进行采样
            log_prob = action_dist.log_prob(action_index)  #log_prob returns the log of the probability density/mass function evaluated at the given sample value.
            prob = torch.exp(log_prob)

            # 这里是暂时先放4个

            # action_space = torch.tensor([[80,90,100,110], [7,3,5,1], [70,40,80,60], [40,20,35,16]])
            action = self.architecture_map[k][action_index.item()]
            regression_params[k] = {}
            regression_params[k]["action"] = action
            total_log_prob = total_log_prob + log_prob
            regression_params[k]["log_prob"] = log_prob.item()
            regression_params[k]["prob"] = prob.item()

        print(regression_params)

        # regression_params = {
        #     "conv1d_out_channels":action[0],
        #     "conv1d_kernel_size":action[1],
        #     "linear1200_80_out_features":action[2],
        #     "linear80_40_out_features":action[3],
        #     "linear40_40_out_features":action[4],
        #
        # }
        return regression_params,total_log_prob








class Task():

    def __init__(self,train_set,val_set):
        self.train_set = train_set
        self.val_set = val_set
        self.num_epoch = 100
        self.train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=True)
        self.loss = nn.MSELoss()  # 所以 loss 使用 MSELoss

    def get_state(self):
        all_train_data_x,all_train_data_y = next(iter(self.train_loader))
        return all_train_data_x#.to(device)


    def train(self,model,train_loader):


        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # optimizer 使用 Adam
        num_epoch = 70

        # 一个epoch指代所有的数据送入网络中完成一次前向计算及反向传播的过程
        for epoch in range(num_epoch):
            train_loss = 0.0
            count = math.ceil(len(train_loader.dataset.indices) / train_loader.batch_size)
            model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
            # 所谓iterations就是完成一次epoch所需的batch个数。
            for i, data in enumerate(train_loader):  # 这里的的data就是 batch中的x和y   enumerate就是把list中的值分成（下标,值）
                optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零

                # print(j)
                # input = data[0].unsqueeze(0)
                train_pred = model(
                    data[0].to(device=device))  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數  input (72,3,128,128)
                batch_loss = self.loss(train_pred, data[1].to(
                    device=device))  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上） groud truth - train_pred
                batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
                # print(str(i))
                optimizer.step()  # 以 optimizer 用 gradient 更新參數值

                # train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())#和groud thuth 比较看正确率
                train_loss += batch_loss.item()

            epoch_loss = train_loss / count
            # print("Epoch :", epoch, "train_loss:", epoch_loss)

        return epoch_loss

    def evaluate(self,model, dataloader):
        model.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for feature, target in dataloader:
                feature, target = feature.to(device), target.to(device)
                output = model(feature)
                loss = self.loss(output, target)
                epoch_loss += loss.item()
        return epoch_loss / len(dataloader)








def evaluate(model, loss_fn, dataloader, device):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for feature, target in dataloader:
            feature, target = feature.to(device), target.to(device)
            output = model(feature)
            loss = loss_fn(output, target)
            epoch_loss += loss.item()
    return epoch_loss/len(dataloader)


def train_one_epoch(model, loss_fn, dataloader,num_epoch,optimizer, device):


    train_loss = 0.0
    count = math.ceil(len(train_x)/batch_size)
    model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
    # 所谓iterations就是完成一次epoch所需的batch个数。
    for i, data in enumerate(dataloader):#这里的的data就是 batch中的x和y   enumerate就是把list中的值分成（下标,值）
        optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零

        # print(j)
        # input = data[0].unsqueeze(0)
        train_pred = model(data[0].to(device=device))  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數  input (72,3,128,128)
        batch_loss = loss_fn(train_pred, data[1].to(device=device))  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上） groud truth - train_pred
        batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
        # print(str(i))
        optimizer.step()  # 以 optimizer 用 gradient 更新參數值

        # train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())#和groud thuth 比较看正确率
        train_loss += batch_loss.item()


    print("train_loss:",train_loss/count)



def train(model, loss_fn, dataloader,num_epoch,optimizer, device):

    # 一个epoch指代所有的数据送入网络中完成一次前向计算及反向传播的过程
    for epoch in range(num_epoch):
        train_loss = 0.0
        count = math.ceil(len(train_x)/batch_size)
        model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
        # 所谓iterations就是完成一次epoch所需的batch个数。
        for i, data in enumerate(dataloader):#这里的的data就是 batch中的x和y   enumerate就是把list中的值分成（下标,值）
            optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零

            # print(j)
            # input = data[0].unsqueeze(0)
            train_pred = model(data[0].to(device=device))  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數  input (72,3,128,128)
            batch_loss = loss_fn(train_pred, data[1].to(device=device))  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上） groud truth - train_pred
            batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
            # print(str(i))
            optimizer.step()  # 以 optimizer 用 gradient 更新參數值

            # train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())#和groud thuth 比较看正确率
            train_loss += batch_loss.item()


        print("Epoch :", epoch ,"train_loss:",train_loss/count)

def reinforcementlearning_main():
    '''
    ## 訓練 Agent

    現在我們開始訓練 agent。
    透過讓 agent 和 environment 互動，我們記住每一組對應的 log probabilities 及 reward，並在成功登陸或者不幸墜毀後，回放這些「記憶」來訓練 policy network。
    '''
    total_train_size = 12000
    EPISODE_PER_BATCH = 1  # 每蒐集 5 個 episodes 更新一次 agent
    task_size = EPISODE_PER_BATCH
    task_data_size = int(total_train_size/task_size)
    # task_data_train = [task_data_size for i in range(task_size)]


    #
    # train_set_list = torch.utils.data.random_split(train_set,
    #                                                task_data_train)
    #
    # val_set_list = torch.utils.data.random_split(val_set, [300, 300, 300, 300, 300, 300, 300, 300, 300, 299])

    # 最後，建立一個 network 和 agent，就可以開始進行訓練了。
    # tp_size = train_x_for_torch[0].shape
    # tp_size = tp_size + (task_data_size,)
    # network = PolicyGradientNetwork(tp_size)
    agent = PolicyGradientAgent()

    agent.network.train()  # 訓練前，先確保 network 處在 training 模式

    NUM_BATCH = 7600  # 總共更新 400 次

    avg_total_rewards, avg_final_rewards = [], []

    for batch in range(NUM_BATCH):
        train_set, val_set = torch.utils.data.random_split(all_train_set, [12000, 2999])
        task_list = []
        for i in range(EPISODE_PER_BATCH):
            task = Task(train_set, val_set)
            task_list.append(task)

        log_probs, rewards = [], []
        total_rewards, final_rewards = [], []


        for episode in range(EPISODE_PER_BATCH):
            task = task_list[episode]
            state = task.get_state()
            total_reward, total_step = 0, 0


            actionparam,log_prob = agent.sample(state)
            model = Regression(actionparam).to(device)
            tr_load = DataLoader(train_set, batch_size=512, shuffle=False)
            val_load = DataLoader(val_set, batch_size=512, shuffle=False)
            action_loss = task.train(model,tr_load)
            evaluate_loss = task.evaluate(model,val_load)
            #  以前main函数训练的结果记为baseline  reward 基于 baseline 来
            mean_loss = (action_loss+evaluate_loss)/2
            reward = 480 - mean_loss
            print("reward:",reward,"mean_loss:",mean_loss," action_loss",action_loss,"evaluate_loss",evaluate_loss)
            agent.learn(reward,log_prob)








def PREPROCESS(lines):
    data_n = len(lines) - 1
    SEQ = zeros((data_n, 34, 4), dtype=int)
    CA = zeros((data_n, 1), dtype=int)

    for l in range(1, data_n + 1):
        data = lines[l].split()
        seq = data[1]
        for i in range(34):
            if seq[i] in "Aa":
                SEQ[l - 1, i, 0] = 1
            elif seq[i] in "Cc":
                SEQ[l - 1, i, 1] = 1
            elif seq[i] in "Gg":
                SEQ[l - 1, i, 2] = 1
            elif seq[i] in "Tt":
                SEQ[l - 1, i, 3] = 1
        CA[l - 1, 0] = int(data[2])

    return SEQ, CA





if __name__ == '__main__':
    # main()
    reinforcementlearning_main()

