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



train_set,test_set = data_load_xcas()

# train_x, train_y, test_x, test_y = data_load()
# # 维度互换
# train_x_for_torch = np.transpose(train_x,(0,2,1))
# test_x__for_torch = np.transpose(test_x,(0,2,1))
# all_train_set = RNADataset(train_x_for_torch,train_y)
# test_set = RNADataset(test_x__for_torch,test_y)
batch_size = 256
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)



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







def evaluate(model, loss_fn, dataloader, device):
    model.eval()
    epoch_loss = 0.0
    dftotal = pd.DataFrame(columns=["bp","predict","ground truth"])
    with torch.no_grad():
        for feature, target in dataloader:
            feature, target = feature.to(device), target.to(device)
            output = model(feature)
            seqs = one_hot_decode(feature)
            df = pd.DataFrame(columns=["bp","predict","ground truth"])
            df["bp"] = seqs
            df["predict"] = output.cpu()
            df["ground truth"] = target.cpu()

            col_truth = []
            # i = 0
            for i, item in enumerate(df["ground truth"]):
                if item >= 70:
                    col_truth.append(1)
                else:
                    col_truth.append(0)

            df["ground truth classification"] = col_truth


            loss = loss_fn(output, target)
            epoch_loss += loss.item()

            dftotal = dftotal.append(df)
    return epoch_loss/len(dataloader),dftotal


def train_one_epoch(model, loss_fn, dataloader,num_epoch,optimizer, device):
    train_loss = 0.0
    count = math.ceil(len(dataloader.dataset)/batch_size)
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



def main():
    # for param in model.parameters():
    #     print(param.data)

    model = Regression().to(device)
    loss = nn.MSELoss()  # 所以 loss 使用 MSELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # optimizer 使用 Adam
    num_epoch = 60

    # train(model, loss, train_loader, num_epoch, optimizer, device)
    # print(evaluate(model,loss,test_loader,device))
    # 一个epoch指代所有的数据送入网络中完成一次前向计算及反向传播的过程
    for epoch in range(num_epoch):
        print("epoch:",epoch)
        train_one_epoch(model, loss, train_loader, num_epoch, optimizer, device)
        avgloss,df = evaluate(model, loss, test_loader, device)
        print(avgloss)
        rho, p = spearmanr(df["predict"],df["ground truth"])
        prho, pp = pearsonr(df["predict"], df["ground truth"])

        fpr, tpr, thresholds = metrics.roc_curve(df["ground truth classification"], df["predict"])
        # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
        roc_auc = metrics.auc(fpr, tpr)

        print("spearman :"+ str(rho))
        print("p :"+str(p))
        print("pearson :"+ str(prho))
        print("pp :"+str(p))
        print("roc_auc :" + str(roc_auc))

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
    main()
    # reinforcementlearning_main()

