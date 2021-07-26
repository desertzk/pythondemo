
import numpy as np

import torch
import torch.nn as nn

import pandas as pd
from numpy import zeros

from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.distributions import Categorical

import math

import torch.nn.functional as F
import logging
from scipy.stats import spearmanr


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debugpadding5.log"),
        logging.StreamHandler()
    ]
)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
logging.info(device)
# cudnn.benchmark = True


def one_hot_decode(feature):
    # data_n = len(feature)
    seqs = []
    for item in feature:
        oseqs = item.cpu().numpy()
        oseqs = np.transpose(oseqs,(1,0))
        seq=""
        for oseq in oseqs:
            if oseq[0] == 1.0:
                seq+="A"
            elif oseq[1] == 1.0:
                seq += "C"
            elif oseq[2] == 1.0:
                seq += "G"
            elif oseq[3] == 1.0:
                seq += "T"
        seqs.append(seq)

    return seqs


def PREPROCESS_ONE_HOT(train_data,mer):
    data_n = len(train_data)
    SEQ = zeros((data_n, mer, 4), dtype=int)
    # CA = zeros((data_n, 1), dtype=int)

    for l in range(0, data_n):
        seq = train_data[l]
        for i in range(mer):
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


def data_load_nc():
    train_data = pd.read_csv('data/DataS1.csv')
    # test_data = pd.read_excel('data/41587_2018_BFnbt4061_MOESM39_ESM.xlsx', sheet_name=1)
    # use_data = train_data[1:14999]
    # new_header = test_data.iloc[0]
    # test_data = test_data[1:]
    # test_data.index = np.arange(0, len(test_data))
    # test_data.columns = new_header
    bp34_col = train_data["21mer"]
    wt_efficiency = train_data["Wt_Efficiency"]
    eSpCas = train_data["eSpCas 9_Efficiency"]
    SpCas9_HF1 = train_data["SpCas9-HF1_Efficiency"]
    # SEQ = PREPROCESS_ONE_HOT(bp34_col,mer=21)

    wt_df = pd.DataFrame({"21mer":bp34_col,"Wt_Efficiency":wt_efficiency})
    wt_df_clean = wt_df.dropna()
    wt_df_clean = wt_df_clean.reset_index(drop=True)

    eSpCas_df = pd.DataFrame({"21mer":bp34_col,"eSpCas 9_Efficiency":eSpCas})
    eSpCas_df_clean = eSpCas_df.dropna()
    eSpCas_df_clean = eSpCas_df_clean.reset_index(drop=True)

    SpCas9_HF1_df = pd.DataFrame({"21mer":bp34_col,"HF1_Efficiency":SpCas9_HF1})
    SpCas9_HF1_df_clean = SpCas9_HF1_df.dropna()
    SpCas9_HF1_df_clean = SpCas9_HF1_df_clean.reset_index(drop=True)


    # test_bp34 = test_data["34 bp synthetic target and target context sequence(4 bp + PAM + 23 bp protospacer + 3 bp)"]
    # test_indel_f = test_data["Indel freqeuncy(Background substracted, %)"]
    # test_SEQ = PREPROCESS_ONE_HOT(test_bp34)
    return wt_df_clean,eSpCas_df_clean,SpCas9_HF1_df_clean

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

wt_df_clean,eSpCas_df_clean,SpCas9_HF1_df_clean = data_load_nc()

wt_train_x = wt_df_clean["21mer"]
wt_efficiency = wt_df_clean["Wt_Efficiency"]
wt_train_x = PREPROCESS_ONE_HOT(wt_train_x,21)
wt_train_x_for_torch = np.transpose(wt_train_x,(0,2,1))
# test_x__for_torch = np.transpose(test_x,(0,2,1))
wt_efficiency_set = RNADataset(wt_train_x_for_torch,wt_efficiency)

eSpCas_train_x = eSpCas_df_clean["21mer"]
eSpCas_efficiency = eSpCas_df_clean["eSpCas 9_Efficiency"]
eSpCas_train_x = PREPROCESS_ONE_HOT(eSpCas_train_x,21)
eSpCas_train_x_for_torch = np.transpose(eSpCas_train_x,(0,2,1))
eSpCas_set = RNADataset(eSpCas_train_x_for_torch,eSpCas_efficiency)


SpCas9_HF1_train_x = SpCas9_HF1_df_clean["21mer"]
SpCas9_HF1_efficiency = SpCas9_HF1_df_clean["HF1_Efficiency"]
SpCas9_HF1_train_x = PREPROCESS_ONE_HOT(SpCas9_HF1_train_x,21)
SpCas9_HF1_train_x_for_torch = np.transpose(SpCas9_HF1_train_x,(0,2,1))
SpCas9_HF1_set = RNADataset(SpCas9_HF1_train_x_for_torch,SpCas9_HF1_efficiency)

wt_train_set,wt_test_set = torch.utils.data.random_split(wt_efficiency_set, [42537+4726, 8341])
eSpCas_train_set,eSpCas_test_set = torch.utils.data.random_split(eSpCas_set, [44842+4982,8793])
# 维度互换
# train_x_for_torch = np.transpose(train_x,(0,2,1))
#
# wt_efficiency_set = RNADataset(train_x,wt_efficiency)
# eSpCas_set = RNADataset(train_x,eSpCas)
# SpCas9_HF1_set = RNADataset(train_x,SpCas9_HF1)
#
batch_size = 512

train_loader = DataLoader(wt_train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(wt_test_set, batch_size=batch_size, shuffle=False)

activation_functions = {
                'Sigmoid': nn.Sigmoid(),
                'Tanh': nn.Tanh(),
                'ReLU': nn.ReLU(),
                'LeakyReLU': nn.LeakyReLU(),
                'ELU': nn.ELU(),
                'Hardshrink': nn.Hardshrink(),
                'Hardswish': nn.Hardswish(),
                'ReLU6': nn.ReLU6(),
                'PReLU': nn.PReLU(),
                'None': nn.Identity()
            }

pooling_funtion = {
    'avg':nn.AvgPool1d,
    'max':nn.MaxPool1d,
    'none':nn.Identity
}

def hook_layer(name):
    def hook(model, input, output):
        logging.info("in hook")
        # activation[name] = output.detach()
    return hook


class Regression(nn.Module):
    #(None,34,4)
    # 这样的params无法精确到每一层各个参数的好坏  只能决定一层的 conv1d linear 好坏  是决定每一层的好坏的
    def __init__(self,dict_params):
        super(Regression, self).__init__()
        # 这里有1200 4个小数
        layers = []
        last_out = 4 # first time
        conv_num = dict_params["old_conv_num"]
        linear_num = dict_params["old_linear_num"]
        dict_params.pop("old_conv_num")
        dict_params.pop("old_linear_num")
        last_input = 21
        # for k,params in dict_params.items():
        for layer_no in range(conv_num):
            # for k, v in params.items():
            #     arr = k.split("@")
            #     k_str = arr[0]
            #     layer_no = arr[1]
            conv1d_out_channels = dict_params.get("conv1d_out_channels@"+str(layer_no)).get("action")
            conv1d_kernel_size = dict_params.get("conv1d_kernel_size@"+str(layer_no)).get("action")
            conv_batchnorm = dict_params.get("conv_batch_norm@" + str(layer_no)).get("action")
            conv_active = dict_params.get("conv_active@"+str(layer_no)).get("action")
            pool_type = dict_params.get("pool_type@"+str(layer_no)).get("action")
            conv_pool = dict_params.get("conv_pool@"+str(layer_no)).get("action")
            conv_dropout = dict_params.get("conv_dropout@"+str(layer_no)).get("action")


            # self.need_pool = params.get("need_pool").get("action")

            # logging.info("model conv1d_out_channels:",conv1d_out_channels," conv1d_kernel_size:",conv1d_kernel_size," linear1200_80_out_features:",linear1200_80_out_features
            #       ,"linear80_40_out_features:",linear80_40_out_features," linear40_40_out_features:",linear40_40_out_features," need_pool:",self.need_pool)
            padding = dict_params.get("conv_padding@"+str(layer_no)).get("action")
            layers.append(nn.Conv1d(last_out, conv1d_out_channels, conv1d_kernel_size, 1,padding = padding,padding_mode="replicate"))
            # MultiheadAttention(d_model, nhead, dropout=dropout)

            if conv_batchnorm is 1:
                layers.append(nn.BatchNorm1d(conv1d_out_channels))
            # after_conv padding 以后大小一样 在kernel =1 时不一样
            last_input = (last_input + 2 * padding - conv1d_kernel_size + 1)
            # self.__setattr__("conv1d@"+str(layer_no),nn.Conv1d(4, conv1d_out_channels, conv1d_kernel_size, 1))
            layers.append(activation_functions[conv_active])
            # self.__setattr__("conv_active@"+str(layer_no),activation_functions[conv_active])
            if pool_type == "avg":
                layers.append(nn.AvgPool1d(conv_pool,1))
            elif pool_type == "max":
                layers.append(nn.MaxPool1d(conv_pool,1))
            else:
                conv_pool = 0



            # self.__setattr__("conv_pool@"+str(layer_no),nn.AvgPool1d(conv_pool))  # size of window 2  (15,80)
            layers.append(nn.Dropout(p=conv_dropout))
            # self.__setattr__("conv_dropout@" + str(layer_no), nn.Dropout(p=conv_dropout))
            # layers.append(nn.Flatten())
            # self.flatten = nn.Flatten()
            padding = 0
            # after pooling
            if conv_pool is not 0:
                last_input = (last_input+2*padding - conv_pool + 1)

            last_out = conv1d_out_channels
            # self.__setattr__("linear@" + str(layer_no),nn.Linear(conv1d_out_channels * input, linear1200_80_out_features))
            # conv1d_kernel_size这里要注意这个conv1d_kernel_size要是奇数不然会有小数点

            # self.linear1200_80 =
            # self.linear80_40 = nn.Linear(linear1200_80_out_features, linear80_40_out_features)  #(None, 40)
            # self.linear40_40 = nn.Linear(linear80_40_out_features, linear40_40_out_features)  # (None, 40)

        layers.append(nn.Flatten())
        # if conv_pool is not 0:
        #     input = int((last_input - conv1d_kernel_size + 1) / conv_pool)
        # else:
        #     input = (last_input - conv1d_kernel_size + 1)

        last_out = conv1d_out_channels * last_input
        if last_out <= 0:
            raise ValueError("input < 0")




        for layer_no in range(linear_num):
            linear_out = dict_params.get("linear_out@"+str(layer_no)).get("action")
            linear_batchnorm = dict_params.get("linear_batch_norm@"+str(layer_no)).get("action")
            linear_active = dict_params.get("linear_active@"+str(layer_no)).get("action")
            linear_dropout = dict_params.get("linear_dropout@"+str(layer_no)).get("action")
            # self.need_pool = params.get("need_pool").get("action")
            # self.__setattr__("conv_dropout@" + str(layer_no), nn.Dropout(p=conv_dropout))
            # self.flatten = nn.Flatten()

            layers.append(nn.Linear(last_out, linear_out))
            if linear_batchnorm is 1:
                layers.append(nn.BatchNorm1d(linear_out))
            layers.append(activation_functions[linear_active])
            layers.append(nn.Dropout(p=linear_dropout))
            last_out = linear_out
        self.sequential = nn.Sequential(*layers)
        self.linear_1 = nn.Linear(last_out, 1)  # (None, 40)



    def forward(self, x):
        out = self.sequential(x)
        out = self.linear_1(out)
        # outconv1d = self.conv1d(x) # 进去4通道出来80通道 (30,80)
        # outact = self.relu(outconv1d)
        # # Seq_deepCpf1_C1 = Convolution1D(80, 5)(Seq_deepCpf1_Input_SEQ)
        # if self.need_pool is 1:
        #     outavg1d = self.avg1d(outact)  # size of window 2  (15,80)
        # else:
        #     outavg1d = outact
        # # Seq_deepCpf1_P1 = AveragePooling1D(2)(Seq_deepCpf1_C1)
        # out_flatten = self.flatten(outavg1d)
        # # Seq_deepCpf1_F = Flatten()(Seq_deepCpf1_P1)

        # out_dropout = self.dropout(out_flatten)
        # # Seq_deepCpf1_DO1 = Dropout(0.3)(Seq_deepCpf1_F)
        # out_linear1200_80 = self.linear1200_80(out_dropout)
        # out_act_linear1200_80 = self.relu(out_linear1200_80)
        # # Seq_deepCpf1_D1 = Dense(80, activation='relu')(Seq_deepCpf1_DO1)
        # out_dropout1200_80 = self.dropout(out_act_linear1200_80)
        # # Seq_deepCpf1_DO2 = Dropout(0.3)(Seq_deepCpf1_D1)
        # out_linear80_40 = self.linear80_40(out_dropout1200_80)
        # out_act80_40 = self.relu(out_linear80_40)
        # # Seq_deepCpf1_D2 = Dense(40, activation='relu')(Seq_deepCpf1_DO2)
        # out_dropout80_40 = self.dropout(out_act80_40)
        # # Seq_deepCpf1_DO3 = Dropout(0.3)(Seq_deepCpf1_D2)
        # out_linear40_40 = self.linear40_40(out_dropout80_40)
        # out_act40_40 = self.relu(out_linear40_40)
        # # Seq_deepCpf1_D3 = Dense(40, activation='relu')(Seq_deepCpf1_DO3)
        # out_dropout40_40 = self.dropout(out_act40_40)
        # # Seq_deepCpf1_DO4 = Dropout(0.3)(Seq_deepCpf1_D3)
        # out = self.linear40_1(out_dropout40_40)
        # Seq_deepCpf1_Output = Dense(1, activation='linear')(Seq_deepCpf1_DO4)
        result = out.squeeze(1)
        return result




# 这里有个问题是state怎么搞
# 我的想法是batch里每个输入的atcg碱基作为state
'''

'''
class PolicyGradientNetwork(nn.Module):

    def __init__(self,architecture_map=None,hidden_size=64):
        super().__init__()
        self.architecture_map = architecture_map

        self.nhid = hidden_size
        self.hidden = self.init_hidden()
        self.flatten = nn.Flatten()

        # self.linear_num_layer = linear_layer
        # 暂时不用数据来当state
        # state_size = 1
        # for i in state:
        #     state_size = state_size * i
        state_size = hidden_size
        self.max_conv_layer = len(architecture_map["conv_num"])
        self.max_linear_layer=len(architecture_map["linear_num"])
        self.conv_layer = nn.Linear(state_size,self.max_conv_layer)
        self.linear_layer = nn.Linear(state_size, self.max_linear_layer)
        self.lstm1 = nn.LSTMCell(state_size, hidden_size)
        self.struct_map = {}
        self.c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)
        # h_t = state
        self.h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)




        for i in range(self.max_conv_layer):
            for k, v in self.architecture_map["conv"].items():
                k_str = k + "@" + str(i)
                self.__setattr__(k_str, nn.Linear(state_size, len(v)))
                # linear = nn.Linear(16, len(v))
                # self.struct_map[k] = linear

        for i in range(self.max_linear_layer):
            for k, v in self.architecture_map["linear"].items():
                k_str = k + "@" + str(i)
                self.__setattr__(k_str, nn.Linear(state_size, len(v)))
                # linear = nn.Linear(16, len(v))
                # self.struct_map[k] = linear



    def forward(self, state,conv_layer=2,linear_layer=2):
        # flt = self.flatten(state)
        # action_prob_list = []
        # hid1 = torch.tanh(self.fc1(flt))
        # hid2 = torch.tanh(self.fc2(hid1))
        action_prob_map = {}
        action_prob_map["conv"] = {}
        action_prob_map["linear"] = {}
        c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)
        h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)

        for i in range(conv_layer):
            # input_data = self.embedding(step_data)
            h_t, c_t = self.lstm1(state, (h_t, c_t))
            # Add drop out
            # h_t = self.drop(h_t)
            # output = self.decoder(h_t)

            for k, v in self.architecture_map["conv"].items():
                k_str = k + "@" + str(i)
                linear_func = self.__getattr__(k_str)
                hid = self.h_t.squeeze()
                result = linear_func(hid)
                result_softmax = F.softmax(result, dim=-1)
                # action_prob = torch.sum(result_softmax, dim=0)
                action_prob_map["conv"][k_str] = result_softmax



        for i in range(linear_layer):
            # input_data = self.embedding(step_data)
            h_t, c_t = self.lstm1(state, (h_t, c_t))
            # Add drop out
            # h_t = self.drop(h_t)
            # output = self.decoder(h_t)

            for k, v in self.architecture_map["linear"].items():
                k_str = k + "@" + str(i)
                linear_func = self.__getattr__(k_str)
                hid = h_t.squeeze()
                result = linear_func(hid)
                result_softmax = F.softmax(result, dim=-1)
                # action_prob = torch.sum(result_softmax, dim=0)
                action_prob_map["linear"][k_str] = result_softmax

        h_t, c_t = self.lstm1(state, (h_t, c_t))
        hid = h_t.squeeze()
        result = self.conv_layer(hid)
        conv_num_softmax = F.softmax(result, dim=-1)
        action_prob_map["conv_num"] = conv_num_softmax

        h_t, c_t = self.lstm1(state, (h_t, c_t))
        hid = h_t.squeeze()
        result = self.linear_layer(hid)
        linear_num_softmax = F.softmax(result, dim=-1)
        action_prob_map["linear_num"] = linear_num_softmax

        return action_prob_map,h_t


    def init_hidden(self):
        h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)
        c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)

        return (h_t, c_t)








# controller
class PolicyGradientAgent():

    def __init__(self):
        # 这样的params无法精确到每一层各个参数的好坏  只能决定一层的 conv1d linear 好坏
        self.layer = 2
        self.conv_num = 2
        self.linear_num = 2

        self.activation_functions_list = ['Sigmoid','Tanh','ReLU','LeakyReLU','ELU','Hardswish','ReLU6','PReLU','None']
        self.pool_type_list = ['avg','max','none']
        self.pool_kernel_list = [2,3,4]
        self.padding_list = [0,1]
        self.conv1d_out_channels_list = [i for i in range(921) if i>9 and i%10==0] #[80,90,100,110]
        self.conv1d_kernel_size_list = [1,3,5,7,9]  #[7,5,1]
        self.drop_out_list = [0,0.05,0.1,0.15,0.2,0.07,0.12,0.16,0.23,0.25,0.3]
        self.conv_num_list = [1,2,3,4]
        self.linear_num_list = [1,2,3,4]
        self.conv_batch_norm_list = [0,1]
        self.batch_norm_list = [0, 1]

        self.linear40_40_out_features_list = [i for i in range(900) if i>2 and i%10==0]#[10,20,40,80]
        self.need_pool = [0,1]



        self.architecture_map ={
            "conv":{
                "conv1d_out_channels": self.conv1d_out_channels_list,
                "conv1d_kernel_size": self.conv1d_kernel_size_list,
                "conv_batch_norm":self.conv_batch_norm_list,
                "conv_active": self.activation_functions_list,
                "conv_padding": self.padding_list,
                "conv_pool": self.pool_kernel_list,
                "pool_type":self.pool_type_list,
                "conv_dropout": self.drop_out_list,
            },
            "conv_num":self.conv_num_list,
            "linear":{
                "linear_out": self.linear40_40_out_features_list,
                "linear_batch_norm": self.batch_norm_list,
                "linear_active": self.activation_functions_list,
                "linear_dropout": self.drop_out_list
            },
            "linear_num": self.linear_num_list,
        }

        self.action_list = []

        self.network = PolicyGradientNetwork(self.architecture_map).to(device)


        self.optimizer = optim.SGD(self.network.parameters(), lr=0.0005)

    def set_new_num(self,new_conv_num,new_linear_num):
        self.conv_num = new_conv_num
        self.linear_num = new_linear_num




    def learn(self, rewards,log_prob):
        # 损失函数要是一个式子
        loss = -torch.mean(log_prob)*rewards
        print("reinfor loss "+str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # for name, param in self.network.named_parameters():
        #     if param.requires_grad:
        #         logging.info(name, param.data)


    def sample_work(self, state):
        item_action_prob = self.network(torch.FloatTensor(state))
        # action_prob = torch.sum(item_action_prob,dim=0)   # To sum over all rows (i.e. for each column)  size = [1, ncol]
        action_dist = Categorical(item_action_prob)
        action_index = action_dist.sample() #.unsqueeze(1)  这里就是根据概率进行采样
        log_prob = action_dist.log_prob(action_index)
        prob = torch.exp(log_prob)
        # 这里是暂时先放4个
        # action_space = torch.tensor([self.conv1d_out_channels_list, self.conv1d_kernel_size_list, self.linear1200_80_out_features_list, self.linear80_40_out_features_list], device=device)
        action_space = torch.tensor([[80,90,100,110], [7,3,5,1], [70,40,80,60], [40,5,35,16]])
        action = torch.gather(action_space, 1, action_index.unsqueeze(1)).squeeze(1)
        # 按照传入的action_prob中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引。
        regression_params = {
            "conv1d_out_channels":{"action":action[0].item(),"log_prob":log_prob[0].item(),"prob":prob[0].item()},
            "conv1d_kernel_size":{"action":action[1].item(),"log_prob":log_prob[1].item(),"prob":prob[1].item()},
            "linear1200_80_out_features":{"action":action[2].item(),"log_prob":log_prob[2].item(),"prob":prob[2].item()},
            "linear80_40_out_features":{"action":action[3].item(),"log_prob":log_prob[3].item(),"prob":prob[3].item()},
            "linear40_40_out_features": {"action":40},
            "need_pool": {"action":1},
        }
        logging.info(regression_params)
        return regression_params,log_prob


    def sample_action(self,action_prob_map,regression_params,key=""):
        for k,item_action_prob in action_prob_map.items():
            if isinstance(item_action_prob,dict):

                self.sample_action(item_action_prob,regression_params,k)

            else:
                action_dist = Categorical(item_action_prob)
                action_index = action_dist.sample() #.unsqueeze(1)  这里就是根据概率进行采样
                log_prob = action_dist.log_prob(action_index)  #log_prob returns the log of the probability density/mass function evaluated at the given sample value.
                prob = torch.exp(log_prob)

                arr = k.split("@")
                if len(arr) == 2:
                    k_str = arr[0]
                    layer_no = arr[1]

                    action = self.architecture_map[key][k_str][action_index.item()]
                    # regression_params[key][k] = {}
                    # regression_params[key][k]["action"] = action
                    # total_log_prob = total_log_prob + log_prob
                    # regression_params[key][k]["log_prob"] = log_prob.item()
                    # regression_params[key][k]["prob"] = prob.item()

                else:
                    action = self.architecture_map[k][action_index.item()]

                regression_params[k] = {}
                regression_params[k]["action"] = action
                self.total_log_prob = self.total_log_prob + log_prob
                regression_params[k]["log_prob"] = log_prob.item()
                regression_params[k]["prob"] = prob.item()





    def sample(self, state,conv_num=2,linear_num=2):

        action_prob_map,new_state = self.network(state,self.conv_num,self.linear_num) #torch.cuda.FloatTensor(
        # action_prob = torch.sum(item_action_prob,dim=0)   # To sum over all rows (i.e. for each column)  size = [1, ncol]

        regression_params = {}
        self.total_log_prob = 0
        self.sample_action(action_prob_map,regression_params)


        # for i in range(self.layer):
        #     regression_params[i] = {}


        # 这里注意也要改
        avg_log_prob = self.total_log_prob/len(regression_params)


        # regression_params = {
        #     "conv1d_out_channels":action[0],
        #     "conv1d_kernel_size":action[1],
        #     "linear1200_80_out_features":action[2],
        #     "linear80_40_out_features":action[3],
        #     "linear40_40_out_features":action[4],
        #
        # }


        regression_params["old_conv_num"] = self.conv_num
        regression_params["old_linear_num"] = self.linear_num
        logging.info(regression_params)
        return regression_params,avg_log_prob,new_state





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
        # model.register_forward_hook(hook_layer("11"))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # optimizer 使用 Adam

        num_epoch = 30

        # 一个epoch指代所有的数据送入网络中完成一次前向计算及反向传播的过程
        for epoch in range(num_epoch):
            train_loss = 0.0
            count = math.ceil(len(train_loader.dataset.indices) / train_loader.batch_size)
            model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
            # 所谓iterations就是完成一次epoch所需的batch个数。
            for i, data in enumerate(train_loader):  # 这里的的data就是 batch中的x和y   enumerate就是把list中的值分成（下标,值）
                optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零

                # logging.info(j)
                # input = data[0].unsqueeze(0)
                train_pred = model(
                    data[0].to(device=device))  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數  input (72,3,128,128)
                batch_loss = self.loss(train_pred, data[1].to(
                    device=device))  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上） groud truth - train_pred
                batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
                # logging.info(str(i))
                optimizer.step()  # 以 optimizer 用 gradient 更新參數值

                # train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())#和groud thuth 比较看正确率
                train_loss += batch_loss.item()

            epoch_loss = train_loss / count
            # logging.info("Epoch :", epoch, "train_loss:", epoch_loss)
        return epoch_loss

    def evaluate(self,model, dataloader):
        model.eval()
        epoch_loss = 0.0
        dftotal = pd.DataFrame()
        with torch.no_grad():
            for feature, target in dataloader:
                df = pd.DataFrame()
                feature, target = feature.to(device), target.to(device)
                output = model(feature)
                seqs = one_hot_decode(feature)
                df["bp"] = seqs
                df["predict"] = output.cpu()
                df["ground truth"] = target.cpu()
                loss = self.loss(output, target)
                epoch_loss += loss.item()
                dftotal = dftotal.append(df)
        return epoch_loss / len(dataloader),dftotal








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






def train(model, loss_fn, dataloader,num_epoch,optimizer, device):

    # 一个epoch指代所有的数据送入网络中完成一次前向计算及反向传播的过程
    for epoch in range(num_epoch):
        train_loss = 0.0
        count = math.ceil(len(train_x)/batch_size)
        model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
        # 所谓iterations就是完成一次epoch所需的batch个数。
        for i, data in enumerate(dataloader):#这里的的data就是 batch中的x和y   enumerate就是把list中的值分成（下标,值）
            optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零

            # logging.info(j)
            # input = data[0].unsqueeze(0)
            train_pred = model(data[0].to(device=device))  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數  input (72,3,128,128)
            batch_loss = loss_fn(train_pred, data[1].to(device=device))  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上） groud truth - train_pred
            batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
            # logging.info(str(i))
            optimizer.step()  # 以 optimizer 用 gradient 更新參數值

            # train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())#和groud thuth 比较看正确率
            train_loss += batch_loss.item()


        logging.info("Epoch :", epoch ,"train_loss:",train_loss/count)


max_reward = -9999999
max_spearman = 0

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

    NUM_BATCH = 58600000  # 總共更新 7600 次

    hidden_size = 64
    state = torch.zeros(1, hidden_size, dtype=torch.float, device=device)

    for batch in range(NUM_BATCH):
        # train_set, val_set = torch.utils.data.random_split(all_train_set, [12000, 2999])
        task = Task(wt_train_set,wt_test_set)
        # 暂时先和数据分离开 state和数据无关
        # state = task.get_state()

        actionparam,log_prob,newstate = agent.sample(state)

        new_conv_num = actionparam["conv_num"]
        new_linear_num = actionparam["linear_num"]
        try:
            model = Regression(actionparam).to(device)
            print(model)
            agent.set_new_num(new_conv_num.get("action"),new_linear_num.get("action"))
            tr_load = DataLoader(wt_train_set, batch_size=512, shuffle=False)
            val_load = DataLoader(wt_test_set, batch_size=512, shuffle=False)
            action_loss = task.train(model,tr_load)
            evaluate_loss,df = task.evaluate(model,val_load)
            rho, p = spearmanr(df["predict"], df["ground truth"])
            if math.isnan(rho):
                rho = 0
                p = 0

            #  以前main函数训练的结果记为baseline  reward 基于 baseline 来
            mean_loss = action_loss*0.15+evaluate_loss*0.85
            spearman_reward = rho * 1000
            reward = spearman_reward
            global max_reward
            global max_spearman

            if rho>max_spearman:
                max_spearman=rho
                logging.error("max_spearman:" + str(max_spearman) + " architecture:" + str(actionparam))


            if reward > max_reward:
                max_reward = reward
                logging.error("max_reward:"+ str(max_reward) +" architecture:" + str(actionparam))
                if max_reward >0 :
                    reward = reward
                    # logging.info("new reward:***********************5555")

            if reward > 0:
                # reward = reward * 3
                # logging.info("reward:***********************3333")
                logging.info("reward:"+str(reward)+"mean_loss:"+str(mean_loss)+" action_loss"+str(action_loss)+"evaluate_loss"+str(evaluate_loss)+"spearmanr "+str(rho) + " p "+str(p))

            print("reward:"+str(reward)+"mean_loss:"+str(mean_loss)+" action_loss"+str(action_loss)+"evaluate_loss"+str(evaluate_loss)+"spearmanr "+str(rho) + " p "+str(p))
            agent.learn(reward,log_prob)
        except ValueError:
            logging.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!input <0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            reward = -1700
            agent.learn(reward, log_prob)
        except Exception as ex:
            logging.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Except"+str(ex))
            reward = -1700
            agent.learn(reward, log_prob)








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

