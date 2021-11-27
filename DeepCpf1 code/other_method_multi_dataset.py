import numpy as np
import os
import numpy as np
# import cv2
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
import pandas as pd
from scipy.stats import spearmanr,pearsonr
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.datasets import make_regression


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
    SEQ = PREPROCESS_ONE_HOT(bp34_col,34)

    test_bp34 = test_data["34 bp synthetic target and target context sequence(4 bp + PAM + 23 bp protospacer + 3 bp)"]
    test_indel_f = test_data["Indel freqeuncy(Background substracted, %)"]
    test_SEQ = PREPROCESS_ONE_HOT(test_bp34,34)
    return SEQ,indel_f,test_SEQ,test_indel_f


class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        self.conv1d = nn.Conv1d(4, 80, 5, 1) # 进去4通道出来80通道 (30,80)
        self.relu = nn.ReLU()
        self.avg1d = nn.AvgPool1d(2) # size of window 2  (15,80)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)
        self.linear1200_80 = nn.Linear(80 * 9, 80)
        self.linear80_40 = nn.Linear(80, 40)  #(None, 40)
        self.linear40_40 = nn.Linear(40, 40)  # (None, 40)
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
        all_train_data = self.train_loader.dataset[:]
        return all_train_data


    def train(self):
        model = Regression().to(device)
        loss = nn.MSELoss()  # 所以 loss 使用 MSELoss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # optimizer 使用 Adam
        num_epoch = 85

        # 一个epoch指代所有的数据送入网络中完成一次前向计算及反向传播的过程
        for epoch in range(num_epoch):
            train_loss = 0.0
            count = math.ceil(len(train_x) / batch_size)
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

            print("Epoch :", epoch, "train_loss:", train_loss / count)

device ="cuda"

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

def data_load_hf1():
    train_data = pd.read_csv('_data_/SpCas9-HF1.csv')

    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]

    hf1bp_train_x = PREPROCESS_ONE_HOT(bp34_col, 23)
    hf1bp_train_x_for_torch = np.transpose(hf1bp_train_x, (0, 2, 1))

    hf1_efficiency100 = xcas_efficiency * 100
    cas_efficiency_set = RNADataset(hf1bp_train_x_for_torch, hf1_efficiency100)

    HF1_train_set, HF1_test_set = torch.utils.data.random_split(cas_efficiency_set, [48354, 8534])

    return HF1_train_set,HF1_test_set


def data_load_wt():
    train_data = pd.read_csv('_data_/WT-SpCas9.csv')

    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]

    hf1bp_train_x = PREPROCESS_ONE_HOT(bp34_col, 23)
    hf1bp_train_x_for_torch = np.transpose(hf1bp_train_x, (0, 2, 1))

    hf1_efficiency100 = xcas_efficiency * 100
    cas_efficiency_set = RNADataset(hf1bp_train_x_for_torch, hf1_efficiency100)

    HF1_train_set, HF1_test_set = torch.utils.data.random_split(cas_efficiency_set, [47263, 8341])

    return HF1_train_set,HF1_test_set

def data_load_esp():
    train_data = pd.read_csv('_data_/raw_eSpCas9.csv')

    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]

    hf1bp_train_x = PREPROCESS_ONE_HOT(bp34_col, 23)
    hf1bp_train_x_for_torch = np.transpose(hf1bp_train_x, (0, 2, 1))

    hf1_efficiency100 = xcas_efficiency * 100
    cas_efficiency_set = RNADataset(hf1bp_train_x_for_torch, hf1_efficiency100)

    HF1_train_set, HF1_test_set = torch.utils.data.random_split(cas_efficiency_set, [49824, 8793])

    return HF1_train_set,HF1_test_set



def data_load_sniper():
    train_data = pd.read_csv('_data_/raw_SniperCas.csv')

    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]

    hf1bp_train_x = PREPROCESS_ONE_HOT(bp34_col, 23)
    hf1bp_train_x_for_torch = np.transpose(hf1bp_train_x, (0, 2, 1))

    hf1_efficiency100 = xcas_efficiency * 100
    cas_efficiency_set = RNADataset(hf1bp_train_x_for_torch, hf1_efficiency100)

    HF1_train_set, HF1_test_set = torch.utils.data.random_split(cas_efficiency_set, [30236, 7558])

    return HF1_train_set,HF1_test_set


def data_load_xcas():
    train_data = pd.read_csv('_data_/raw_xCas.csv')

    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]

    hf1bp_train_x = PREPROCESS_ONE_HOT(bp34_col, 23)
    hf1bp_train_x_for_torch = np.transpose(hf1bp_train_x, (0, 2, 1))

    hf1_efficiency100 = xcas_efficiency * 100
    cas_efficiency_set = RNADataset(hf1bp_train_x_for_torch, hf1_efficiency100)

    HF1_train_set, HF1_test_set = torch.utils.data.random_split(cas_efficiency_set, [30191, 7547])

    return HF1_train_set,HF1_test_set


def data_load_spcas9():
    train_data = pd.read_csv('_data_/raw_SpCas9.csv')

    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]

    hf1bp_train_x = PREPROCESS_ONE_HOT(bp34_col, 23)
    hf1bp_train_x_for_torch = np.transpose(hf1bp_train_x, (0, 2, 1))

    hf1_efficiency100 = xcas_efficiency * 100
    cas_efficiency_set = RNADataset(hf1bp_train_x_for_torch, hf1_efficiency100)

    HF1_train_set, HF1_test_set = torch.utils.data.random_split(cas_efficiency_set, [24468, 6117])

    return HF1_train_set,HF1_test_set


def PREPROCESS_TO_NUM(train_data,mer):
    data_n = len(train_data)
    SEQ = zeros((data_n, mer), dtype=int)
    # CA = zeros((data_n, 1), dtype=int)

    for l in range(0, data_n):

        seq = train_data[l]
        for i in range(mer):
            if seq[i] in "Aa":
                SEQ[l, i] = 1
            elif seq[i] in "Cc":
                SEQ[l, i] = 2
            elif seq[i] in "Gg":
                SEQ[l, i] = 3
            elif seq[i] in "Tt":
                SEQ[l, i] = 4
        # CA[l - 1, 0] = int(data[2])

    return SEQ
# train_set,test_set = data_load_xcas()


# X, y = make_regression(n_features=4, n_informative=2,
#                         random_state=0, shuffle=False)
def data_load_spcas9():
    print("spcas9")
    train_data = pd.read_csv('_data_/raw_SpCas9.csv')
    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]
    col_truth = []
    for item in xcas_efficiency:
        if item>0.7:
            col_truth.append(1)
        else:
            col_truth.append(0)
    X = PREPROCESS_TO_NUM(bp34_col,23)
    xtrain = X[0:24468]
    xtest = X[24468:30585]
    ytrain = xcas_efficiency[0:24468]
    ytrain_classify =col_truth[0:24468]
    ytest = xcas_efficiency[24468:30585]
    ytest_classify = col_truth[24468:30585]
    regr = RandomForestRegressor(max_depth=4, random_state=0)
    regr.fit(xtrain, ytrain)
    ypredict = regr.predict(xtest)
    rfrho, p = spearmanr(ypredict, ytest)
    rfprho, pp = pearsonr(ypredict, ytest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest_classify ,ypredict)
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
    roc_auc = metrics.auc(fpr, tpr)
    print("RandomForest spearmanr %f  pearsonr  %f  roc_auc  %f"%(rfrho,rfprho,roc_auc))
    # return
    boostreg = GradientBoostingRegressor(max_depth=4,
                                         n_estimators=30,
                                         learning_rate=0.1)
    boostreg.fit(xtrain, ytrain)
    boostypredict = boostreg.predict(xtest)
    boostrho, p = spearmanr(boostypredict, ytest)
    boostprho, pp = pearsonr(boostypredict, ytest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest_classify,boostypredict )
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
    boostroc_auc = metrics.auc(fpr, tpr)
    print("GradientBoost spearmanr %f  pearsonr  %f  roc_auc  %f"%(boostrho, boostprho, boostroc_auc))

def data_load_xcas():
    print("xcas")
    train_data = pd.read_csv('_data_/raw_xCas.csv')
    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]
    col_truth = []
    for item in xcas_efficiency:
        if item>0.7:
            col_truth.append(1)
        else:
            col_truth.append(0)
    X = PREPROCESS_TO_NUM(bp34_col,23)
    xtrain = X[0:30191]
    xtest = X[30191:30191+7547]
    ytrain = xcas_efficiency[0:30191]
    ytrain_classify =col_truth[0:30191]
    ytest = xcas_efficiency[30191:30191+7547]
    ytest_classify = col_truth[30191:30191+7547]
    regr = RandomForestRegressor(max_depth=4, random_state=0)
    regr.fit(xtrain, ytrain)
    ypredict = regr.predict(xtest)
    rfrho, p = spearmanr(ypredict, ytest)
    rfprho, pp = pearsonr(ypredict, ytest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest_classify ,ypredict)
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
    roc_auc = metrics.auc(fpr, tpr)
    print("RandomForest spearmanr %f  pearsonr  %f  roc_auc  %f"%(rfrho,rfprho,roc_auc))
    # return
    boostreg = GradientBoostingRegressor(max_depth=4,
                                         n_estimators=30,
                                         learning_rate=0.1)
    boostreg.fit(xtrain, ytrain)
    boostypredict = boostreg.predict(xtest)
    boostrho, p = spearmanr(boostypredict, ytest)
    boostprho, pp = pearsonr(boostypredict, ytest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest_classify,boostypredict )
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
    boostroc_auc = metrics.auc(fpr, tpr)
    print("GradientBoost spearmanr %f  pearsonr  %f  roc_auc  %f"%(boostrho, boostprho, boostroc_auc))



def data_load_snipercas():
    print("snipercas")
    train_data = pd.read_csv('_data_/raw_SniperCas.csv')
    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]
    train_num = 30236
    test_num = 7558
    col_truth = []
    for item in xcas_efficiency:
        if item>0.7:
            col_truth.append(1)
        else:
            col_truth.append(0)
    X = PREPROCESS_TO_NUM(bp34_col,23)
    xtrain = X[0:train_num]
    xtest = X[train_num:train_num+test_num]
    ytrain = xcas_efficiency[0:train_num]
    ytrain_classify =col_truth[0:train_num]
    ytest = xcas_efficiency[train_num:train_num+test_num]
    ytest_classify = col_truth[train_num:train_num+test_num]
    regr = RandomForestRegressor(max_depth=4, random_state=0)
    regr.fit(xtrain, ytrain)
    ypredict = regr.predict(xtest)
    rfrho, p = spearmanr(ypredict, ytest)
    rfprho, pp = pearsonr(ypredict, ytest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest_classify ,ypredict)
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
    roc_auc = metrics.auc(fpr, tpr)
    print("RandomForest spearmanr %f  pearsonr  %f  roc_auc  %f"%(rfrho,rfprho,roc_auc))
    # return
    boostreg = GradientBoostingRegressor(max_depth=4,
                                         n_estimators=30,
                                         learning_rate=0.1)
    boostreg.fit(xtrain, ytrain)
    boostypredict = boostreg.predict(xtest)
    boostrho, p = spearmanr(boostypredict, ytest)
    boostprho, pp = pearsonr(boostypredict, ytest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest_classify,boostypredict )
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
    boostroc_auc = metrics.auc(fpr, tpr)
    print("GradientBoost spearmanr %f  pearsonr  %f  roc_auc  %f"%(boostrho, boostprho, boostroc_auc))


def data_load_esp():
    print("esp")
    train_data = pd.read_csv('_data_/raw_eSpCas9.csv')
    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]
    train_num = 49824
    test_num = 8793
    col_truth = []
    for item in xcas_efficiency:
        if item>0.7:
            col_truth.append(1)
        else:
            col_truth.append(0)
    X = PREPROCESS_TO_NUM(bp34_col,23)
    xtrain = X[0:train_num]
    xtest = X[train_num:train_num+test_num]
    ytrain = xcas_efficiency[0:train_num]
    ytrain_classify =col_truth[0:train_num]
    ytest = xcas_efficiency[train_num:train_num+test_num]
    ytest_classify = col_truth[train_num:train_num+test_num]
    regr = RandomForestRegressor(max_depth=4, random_state=0)
    regr.fit(xtrain, ytrain)
    ypredict = regr.predict(xtest)
    rfrho, p = spearmanr(ypredict, ytest)
    rfprho, pp = pearsonr(ypredict, ytest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest_classify ,ypredict)
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
    roc_auc = metrics.auc(fpr, tpr)
    print("RandomForest spearmanr %f  pearsonr  %f  roc_auc  %f"%(rfrho,rfprho,roc_auc))
    # return
    boostreg = GradientBoostingRegressor(max_depth=4,
                                         n_estimators=30,
                                         learning_rate=0.1)
    boostreg.fit(xtrain, ytrain)
    boostypredict = boostreg.predict(xtest)
    boostrho, p = spearmanr(boostypredict, ytest)
    boostprho, pp = pearsonr(boostypredict, ytest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest_classify,boostypredict )
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
    boostroc_auc = metrics.auc(fpr, tpr)
    print("GradientBoost spearmanr %f  pearsonr  %f  roc_auc  %f"%(boostrho, boostprho, boostroc_auc))



def data_load_wt():
    print("wt")
    train_data = pd.read_csv('_data_/WT-SpCas9.csv')
    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]
    train_num = 47263
    test_num = 8341
    col_truth = []
    for item in xcas_efficiency:
        if item>0.7:
            col_truth.append(1)
        else:
            col_truth.append(0)
    X = PREPROCESS_TO_NUM(bp34_col,23)
    xtrain = X[0:train_num]
    xtest = X[train_num:train_num+test_num]
    ytrain = xcas_efficiency[0:train_num]
    ytrain_classify =col_truth[0:train_num]
    ytest = xcas_efficiency[train_num:train_num+test_num]
    ytest_classify = col_truth[train_num:train_num+test_num]
    regr = RandomForestRegressor(max_depth=4, random_state=0)
    regr.fit(xtrain, ytrain)
    ypredict = regr.predict(xtest)
    rfrho, p = spearmanr(ypredict, ytest)
    rfprho, pp = pearsonr(ypredict, ytest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest_classify ,ypredict)
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
    roc_auc = metrics.auc(fpr, tpr)
    print("RandomForest spearmanr %f  pearsonr  %f  roc_auc  %f"%(rfrho,rfprho,roc_auc))
    # return
    boostreg = GradientBoostingRegressor(max_depth=4,
                                         n_estimators=30,
                                         learning_rate=0.1)
    boostreg.fit(xtrain, ytrain)
    boostypredict = boostreg.predict(xtest)
    boostrho, p = spearmanr(boostypredict, ytest)
    boostprho, pp = pearsonr(boostypredict, ytest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest_classify,boostypredict )
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
    boostroc_auc = metrics.auc(fpr, tpr)
    print("GradientBoost spearmanr %f  pearsonr  %f  roc_auc  %f"%(boostrho, boostprho, boostroc_auc))




def data_load_HF1():
    print("hf1")
    train_data = pd.read_csv('_data_/SpCas9-HF1.csv')
    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]
    train_num = 48354
    test_num = 8534
    col_truth = []
    for item in xcas_efficiency:
        if item>0.7:
            col_truth.append(1)
        else:
            col_truth.append(0)
    X = PREPROCESS_TO_NUM(bp34_col,23)
    xtrain = X[0:train_num]
    xtest = X[train_num:train_num+test_num]
    ytrain = xcas_efficiency[0:train_num]
    ytrain_classify =col_truth[0:train_num]
    ytest = xcas_efficiency[train_num:train_num+test_num]
    ytest_classify = col_truth[train_num:train_num+test_num]
    regr = RandomForestRegressor(max_depth=4, random_state=0)
    regr.fit(xtrain, ytrain)
    ypredict = regr.predict(xtest)
    rfrho, p = spearmanr(ypredict, ytest)
    rfprho, pp = pearsonr(ypredict, ytest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest_classify ,ypredict)
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
    roc_auc = metrics.auc(fpr, tpr)
    print("RandomForest spearmanr %f  pearsonr  %f  roc_auc  %f"%(rfrho,rfprho,roc_auc))
    # return
    boostreg = GradientBoostingRegressor(max_depth=4,
                                         n_estimators=30,
                                         learning_rate=0.1)
    boostreg.fit(xtrain, ytrain)
    boostypredict = boostreg.predict(xtest)
    boostrho, p = spearmanr(boostypredict, ytest)
    boostprho, pp = pearsonr(boostypredict, ytest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest_classify,boostypredict )
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
    boostroc_auc = metrics.auc(fpr, tpr)
    print("GradientBoost spearmanr %f  pearsonr  %f  roc_auc  %f"%(boostrho, boostprho, boostroc_auc))

data_load_spcas9()
data_load_xcas()
data_load_snipercas()
data_load_esp()
data_load_wt()
data_load_HF1()
