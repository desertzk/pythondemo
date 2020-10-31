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

device ="cuda"

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


class Regression(nn.Module):
    #(None,34,4)
    def __init__(self,params):
        super(Regression, self).__init__()
        # 这里有1200 4个小数
        conv1d_out_channels = params.get("conv1d_out_channels").item()
        conv1d_kernel_size = params.get("conv1d_kernel_size").item()
        linear1200_80_out_features = params.get("linear1200_80_out_features").item()
        linear80_40_out_features= params.get("linear80_40_out_features").item()
        print("model conv1d_out_channels:",conv1d_out_channels," conv1d_kernel_size:",conv1d_kernel_size," linear1200_80_out_features:",linear1200_80_out_features
              ,"linear80_40_out_features:",linear80_40_out_features)
        self.conv1d = nn.Conv1d(4, conv1d_out_channels, conv1d_kernel_size, 1) # 进去4通道出来80通道 (30,80)
        self.relu = nn.ReLU()
        self.avg1d = nn.AvgPool1d(2) # size of window 2  (15,80)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)
        # conv1d_kernel_size这里要注意这个conv1d_kernel_size要是奇数不然会有小数点
        input = int((34-conv1d_kernel_size+1)/2)
        self.linear1200_80 = nn.Linear(conv1d_out_channels * input, linear1200_80_out_features)
        self.linear80_40 = nn.Linear(linear1200_80_out_features, linear80_40_out_features)  #(None, 40)
        self.linear40_40 = nn.Linear(linear80_40_out_features, 40)  # (None, 40)
        self.linear40_1 = nn.Linear(40, 1)  # (None, 40)



    def forward(self, x):
        outconv1d = self.conv1d(x) # 进去4通道出来80通道 (30,80)
        outact = self.relu(outconv1d)
        # Seq_deepCpf1_C1 = Convolution1D(80, 5)(Seq_deepCpf1_Input_SEQ)
        outavg1d = self.avg1d(outact)  # size of window 2  (15,80)
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
## Policy Gradient

現在來搭建一個簡單的 policy network。
我們預設模型的輸入是 8-dim 的 observation，輸出則是離散的四個動作之一：
'''
class PolicyGradientNetwork(nn.Module):

    def __init__(self,state):
        super().__init__()
        self.flatten = nn.Flatten()
        self.num_param = 4
        state_size = 1
        for i in state:
            state_size = state_size * i
        self.fc1 = nn.Linear(state_size, 64)   # 4*34  64
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)     #最后输出的4dim是4个动作的概率

    def forward(self, state):
        flt = self.flatten(state)
        result_list = []
        for i in range(self.num_param):
            hid = torch.tanh(self.fc1(flt))
            hid = torch.tanh(self.fc2(hid))
            hid3 = self.fc3(hid)
            result = F.softmax(hid3, dim=-1)
            action_prob = torch.sum(result,dim=0)   # To sum over all rows (i.e. for each column)  size = [1, ncol]
            #
            result_list.append(action_prob)

        outputs = torch.stack(result_list).squeeze(1)

        return outputs


'''
再來，搭建一個簡單的 agent，並搭配上方的 policy network 來採取行動。
這個 agent 能做到以下幾件事：
- `learn()`：從記下來的 log probabilities 及 rewards 來更新 policy network。
- `sample()`：從 environment 得到 observation 之後，利用 policy network 得出應該採取的行動。
而此函式除了回傳抽樣出來的 action，也會回傳此次抽樣的 log probabilities。
'''

class PolicyGradientAgent():

    def __init__(self, network):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.conv1d_out_channels_list = [i for i in range(100) if i>9]
        self.conv1d_kernel_size_list = [i for i in range(15) if i>0]
        self.linear1200_80_out_features_list = [i for i in range(1000) if i>9]
        self.linear80_40_out_features_list = [i for i in range(100) if i>9]

    def learn(self, rewards,log_prob):
        # 损失函数要是一个式子
        loss = -torch.mean(log_prob)*rewards

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        item_action_prob = self.network(torch.FloatTensor(state))
        # action_prob = torch.sum(item_action_prob,dim=0)   # To sum over all rows (i.e. for each column)  size = [1, ncol]
        action_dist = Categorical(item_action_prob)
        action_index = action_dist.sample() #.unsqueeze(1)  这里就是根据概率进行采样
        log_prob = action_dist.log_prob(action_index)
        # log_prob = action_dist.log_prob(action_index)
        # mask = one_hot(action_index, num_classes=self.ACTION_SPACE)
        #
        # episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)
        #
        # # append the action to the episode action list to obtain the trajectory
        # # we need to store the actions and logits so we could calculate the gradient of the performance
        # # episode_actions = torch.cat((episode_actions, action_index), dim=0)
        #
        # # Get action actions
        # action_space = torch.tensor([[3, 5, 7], [8, 16, 32], [3, 5, 7], [8, 16, 32]], device=self.DEVICE)
        # action = torch.gather(action_space, 1, action_index).squeeze(1)

        # 这里是暂时先放4个
        # action_space = torch.tensor([self.conv1d_out_channels_list, self.conv1d_kernel_size_list, self.linear1200_80_out_features_list, self.linear80_40_out_features_list], device=device)
        action_space = torch.tensor([[80,90,100,110], [7,3,5,1], [70,40,80,60], [40,20,35,16]])
        action = torch.gather(action_space, 1, action_index.unsqueeze(1)).squeeze(1)
        # 按照传入的action_prob中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引。
        regression_params = {
            "conv1d_out_channels":action[0],
            "conv1d_kernel_size":action[1],
            "linear1200_80_out_features":action[2],
            "linear80_40_out_features":action[3],
        }
        return regression_params,log_prob




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



class Task():

    def __init__(self,train_set,val_set):
        self.train_set = train_set
        self.val_set = val_set
        self.num_epoch = 100
        self.train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=True)


    def get_state(self):
        all_train_data_x,all_train_data_y = next(iter(self.train_loader))
        return all_train_data_x


    def train(self,model,train_loader):

        loss = nn.MSELoss()  # 所以 loss 使用 MSELoss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # optimizer 使用 Adam
        num_epoch = 50

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
                batch_loss = loss(train_pred, data[1].to(
                    device=device))  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上） groud truth - train_pred
                batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
                # print(str(i))
                optimizer.step()  # 以 optimizer 用 gradient 更新參數值

                # train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())#和groud thuth 比较看正确率
                train_loss += batch_loss.item()

            epoch_loss = train_loss / count
            # print("Epoch :", epoch, "train_loss:", epoch_loss)

        return epoch_loss


train_x, train_y, test_x, test_y = data_load()
# 维度互换
train_x_for_torch = np.transpose(train_x,(0,2,1))
test_x__for_torch = np.transpose(test_x,(0,2,1))
all_train_set = RNADataset(train_x_for_torch,train_y)
test_set = RNADataset(test_x__for_torch,test_y)
batch_size = len(all_train_set)






train_loader = DataLoader(all_train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)





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
    EPISODE_PER_BATCH = 10  # 每蒐集 5 個 episodes 更新一次 agent
    task_size = EPISODE_PER_BATCH
    task_data_size = int(total_train_size/task_size)
    task_data_train = [task_data_size for i in range(task_size)]

    train_set, val_set = torch.utils.data.random_split(all_train_set, [12000, 2999])

    train_set_list = torch.utils.data.random_split(train_set,
                                                   task_data_train)

    val_set_list = torch.utils.data.random_split(val_set, [300, 300, 300, 300, 300, 300, 300, 300, 300, 299])

    # 最後，建立一個 network 和 agent，就可以開始進行訓練了。
    tp_size = train_x_for_torch[0].shape
    # tp_size = tp_size + (task_data_size,)
    network = PolicyGradientNetwork(tp_size)
    agent = PolicyGradientAgent(network)

    agent.network.train()  # 訓練前，先確保 network 處在 training 模式

    NUM_BATCH = 600  # 總共更新 400 次

    avg_total_rewards, avg_final_rewards = [], []

    for batch in range(NUM_BATCH):


        task_list = []
        for i in range(EPISODE_PER_BATCH):
            task = Task(train_set_list[i], val_set_list[i])
            task_list.append(task)

        log_probs, rewards = [], []
        total_rewards, final_rewards = [], []


        for episode in range(EPISODE_PER_BATCH):
            task = task_list[episode]
            state = task.get_state()
            total_reward, total_step = 0, 0


            actionparam,log_prob = agent.sample(state)
            model = Regression(actionparam).to(device)
            tr_load = DataLoader(train_set_list[episode], batch_size=128, shuffle=False)
            action_loss = task.train(model,tr_load)
            #  以前main函数训练的结果记为baseline  reward 基于 baseline 来
            reward = 300 - action_loss
            print("reward:",reward," action_loss",action_loss)
            agent.learn(reward,log_prob)
            # action, log_prob = agent.sample(state)
            # next_state, reward, done, _ = env.step(action)
            # log_probs.append(log_prob)
            # state = next_state
            # total_reward += reward
            # total_step += 1

            # if done:
            #     if reward > -200:
            #         img.set_data(env.render(mode='rgb_array'))
            #         display.display(plt.gcf())
            #         display.clear_output(wait=True)
            #     final_rewards.append(reward)
            #
            #     total_rewards.append(total_reward)
            #     rewards.append(
            #         np.full(total_step, total_reward))  # 設定同一個 episode 每個 action 的 reward 都是 total reward
            #     break

        # 紀錄訓練過程
        # avg_total_reward = sum(total_rewards) / len(total_rewards)
        # avg_final_reward = sum(final_rewards) / len(final_rewards)
        # avg_total_rewards.append(avg_total_reward)
        # avg_final_rewards.append(avg_final_reward)
        # print("Batch {},\tTotal Reward = {:.1f},\tFinal Reward = {:.1f}".format(batch + 1, avg_total_reward,
        #                                                                         avg_final_reward))
        #
        # # 更新網路
        # rewards = np.concatenate(rewards, axis=0)
        # rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
        # agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))


def main():
    # for param in model.parameters():
    #     print(param.data)

    model = Regression().to(device)
    loss = nn.MSELoss()  # 所以 loss 使用 MSELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # optimizer 使用 Adam
    num_epoch = 150

    train(model, loss, train_loader, num_epoch, optimizer, device)
    print(evaluate(model,loss,test_loader,device))
    # 一个epoch指代所有的数据送入网络中完成一次前向计算及反向传播的过程
    for epoch in range(num_epoch):
        train_one_epoch(model, loss, train_loader, num_epoch, optimizer, device)
    #     train_loss = 0.0
    #     count = math.ceil(len(train_x)/batch_size)
    #     model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
    #     # 所谓iterations就是完成一次epoch所需的batch个数。
    #     for i, data in enumerate(train_loader):#这里的的data就是 batch中的x和y   enumerate就是把list中的值分成（下标,值）
    #         optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
    #
    #         # print(j)
    #         # input = data[0].unsqueeze(0)
    #         train_pred = model(data[0].to(device=device))  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數  input (72,3,128,128)
    #         batch_loss = loss(train_pred, data[1].to(device=device))  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上） groud truth - train_pred
    #         batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
    #         # print(str(i))
    #         optimizer.step()  # 以 optimizer 用 gradient 更新參數值
    #
    #         # train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())#和groud thuth 比较看正确率
    #         train_loss += batch_loss.item()
    #
    #
    #     print("Epoch :", epoch ,"train_loss:",train_loss/count)

'''

        # evaluate()
        # print(evaluate(model,loss,test_loader,device))
        # model.eval()
        # with torch.no_grad():
        #     for i, data in enumerate(test_loader):
        #         val_pred = model(data[0].to(device="cuda",dtype=torch.float))  #data[0] 图片 data[1] 结果
        #         batch_loss = loss(val_pred, data[1].to(device="cuda",dtype=torch.float))
        #
        #         val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        #         val_loss += batch_loss.item()
        #
        #     # 將結果 print 出來
        #     print("train_loss:"+str(train_loss)+" val loss"+str(val_loss))
        #     print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
        #           (epoch + 1, num_epoch, time.time() - epoch_start_time, \
        #            train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / test_set.__len__(),
        #            val_loss / test_set.__len__()))
        '''






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

