
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
from itertools import islice
import operator
from collections import OrderedDict
from torch._jit_internal import _copy_to_script_wrapper

from typing import Any, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union
from pathlib import Path

T = TypeVar('T')

g_batch_size = 512

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("newwtdebugpadding210803.log"),
        logging.StreamHandler()
    ]
)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
logging.info(device)
# cudnn.benchmark = True
PATH = "eSpCas_undefine_checkpoint_last.pt"

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


def conv_pre_hook1(model, input):
    # In other words, for an input of size (N, C_{in}, L_{in})(N,C
    # in,Lin)
    change_input = input
    if input[0].shape_name[1] == "batch":  # 前面有mutiheadattion 交换了 batch    这里第三轮shape——name没有转对
        change_input = input[0].permute(1, 2, 0)

    elif input[0].shape[2] == model.in_channels:
        change_input = input[0].permute(0, 2, 1)

        change_input = (change_input)
    elif input[0].shape[1]!=model.in_channels:
        #reshape
        print("reshape")
        change_input = input[0].view(model.in_channels,-1)
        change_input = (change_input)
    #print(" conv_pre_hook input ",input[0].shape,change_input[0].shape)
    return change_input

def conv_pre_hook(model, input):
    c_idx = input[0].shape_name.index("hot")
    b_idx = input[0].shape_name.index("batch")
    l_idx = input[0].shape_name.index("len")
    change_input = input[0].permute(b_idx, c_idx, l_idx)
    # print(" conv_pre_hook input ", input[0].shape, change_input.shape)
    return change_input



def other_pre_hook(model, input):
    change_input = input
    if input[0].shape_name[1] == "batch":  #前面有mutiheadattion 交换了 batch
        change_input = input[0].permute(1, 0, 2)
    if type(change_input) is tuple:
        change_input = input[0].flatten(1, -1)
        change_input = (change_input)
    else:
        change_input = change_input.flatten(1, -1)
        change_input = (change_input)
    # print(" other_pre_hook input ", input[0].shape, change_input.shape)
    return change_input


def polling_pre_hook(model, input):
    change_input = input
    #  (N, C, L)(N,C,L) and output (N, C, L_{out})(N,C,L
    # out
    # ​
    #  )

    c_idx = input[0].shape_name.index("hot")
    b_idx = input[0].shape_name.index("batch")
    l_idx = input[0].shape_name.index("len")
    change_input = input[0].permute(b_idx, c_idx, l_idx)
    # print(" polling_pre_hook input ", input[0].shape, change_input.shape)
    return change_input

def polling_after_hook(model, input,result):
    # 如果是前面有conv的需要换位指定 c
    # (N,C,L)
    if input[0].ndim == 3:
        result.shape_name = ("batch","hot", "len")
    return result

def linear_pre_hook(model, input):
    change_input = input
    # change to (batch lenth one_hot）
    # if model.in_features != input[0].shape[-1] and model.in_features == input[0].shape[1]:
    #     change_input = input[0].permute(0, 2, 1)
    #     change_input = (change_input)
    #     change_input.shape_name = ("batch","len", "hot")
    c_idx = input[0].shape_name.index("hot")
    b_idx = input[0].shape_name.index("batch")
    l_idx = input[0].shape_name.index("len")
    change_input = input[0].permute(b_idx, l_idx, c_idx)
    # print(" linear_pre_hook input ", input[0].shape, change_input.shape)
    return change_input

def linear_after_hook(model, input,result):
    # 如果是前面有conv的需要换位指定 c
    # num_features – C from an expected input of size (N, C, L) or L from input of size (N, L)
    if input[0].ndim == 3:
        result.shape_name = ("batch","len", "hot")
    return result

def batch_norm_pre_hook(model, input):
    # 如果是前面有conv的需要换位指定 c
    # num_features – C from an expected input of size (N, C, L) or L from input of size (N, L)
    # change_input = input
    c_idx = input[0].shape_name.index("hot")
    b_idx = input[0].shape_name.index("batch")
    l_idx = input[0].shape_name.index("len")
    change_input = input[0].permute(b_idx, c_idx, l_idx)
    return change_input

def batch_norm_after_hook(model, input,result):
    # 如果是前面有conv的需要换位指定 c
    # num_features – C from an expected input of size (N, C, L) or L from input of size (N, L)
    result.shape_name = ("batch", "hot","len")
    return result


def embedding_pre_hook(model, input):
    input = input[0].long()
    # input = (input2) #这里转换有问题
    change_input = input
    if type(input) is tuple:
        if input[0].shape[1] != model.num_embeddings:
            change_input = input[0].permute(0, 2, 1)  #上一回合 linear 已经交换过了(batch lenth one_hot  位置
            if change_input.shape[1] != model.num_embeddings:
                change_input = input[0].view(input[0].shape[0],model.num_embeddings,-1)
            change_input = (change_input)
        elif input[0].shape[1] == g_batch_size and input[0].shape[2] != model.num_embeddings:
            change_input = input[0].permute(1, 0, 2)  #上一回合 mutihead 已经交换过了(batch lenth one_hot  位置
            if change_input.shape[1] != model.num_embeddings:
                change_input = input[0].view(input[0].shape[0],model.num_embeddings,-1)
            change_input = (change_input)
    else:
        if input.shape[1] != model.num_embeddings:
            change_input = input.permute(0, 2, 1)  # 上一回合 linear 已经交换过了(batch lenth one_hot  位置
            if change_input.shape[1] != model.num_embeddings:
                change_input = input.view(input.shape[0], model.num_embeddings, -1)
            change_input = (change_input)
        elif input.shape[1] == g_batch_size and input.shape[2] != model.num_embeddings:
            change_input = input.permute(1, 0, 2)  # 上一回合 mutihead 已经交换过了(batch lenth one_hot  位置
            if change_input.shape[1] != model.num_embeddings:
                change_input = input.view(input.shape[0], model.num_embeddings, -1)
            change_input = (change_input)
    print(" embedding_pre_hook input ",input.shape,change_input.shape)
    return change_input

def mutiheadattention_pre_hook(model, input):
    '''
    query: (L, N, E)(L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.

key: (S, N, E)(S,N,E) , where S is the source sequence length, N is the batch size, E is the embedding dimension.

value: (S, N, E)(S,N,E) where S is the source sequence length, N is the batch size, E is the embedding dimension.

    '''
    # iinput batch hot4 length
    #change_input  (length, batch size, d_model)

    # if len(input)!=1: #说明调整过了
    #     return input
    # if model.embed_dim == input[0].shape[2]:   #after linear
    #     change_input = input[0].permute(1, 0, 2)
    # elif model.embed_dim == input[0].shape[1]:  #start
    #     change_input = input[0].permute(2, 0, 1)


    c_idx = input[0].shape_name.index("hot")
    b_idx = input[0].shape_name.index("batch")
    l_idx = input[0].shape_name.index("len")
    change_input = input[0].permute(l_idx, b_idx, c_idx)



    input0 = change_input
    input1= change_input
    input2 = change_input
    change_input_tuple = (input0,input1,input2)
    # print("mutiheadattention_pre_hook")
    return change_input_tuple


def mutiheadattention_after_forward(model, input,result):
    '''
    - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.

    '''

    attn_output,attn_output_weights = result
    attn_output.shape_name = ("len", "batch", "hot")
    return attn_output


def conv_after_forward(model, input,result):
    '''


    '''

    # 需要加上不然这个shape——name不能传到下一层
    result.shape_name = ("batch", "hot", "len")
    return result


T = TypeVar('T')

class MySequential(nn.Sequential):
    @overload
    def __init__(self, *args: nn.Sequential) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...

    def __init__(self, *args: Any):
        super(nn.Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    @overload
    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self: T, idx) -> T:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def forward(self, input):
        for module in self:
            # print(input.shape)
            input1 = module(input)
            if not hasattr(input1,"shape_name"):
                input1.shape_name = input.shape_name
            input = input1
        return input1



class Regression(nn.Module):
    #(None,34,4)
    #
    def __init__(self,struct_list):
        super(Regression, self).__init__()
        # 这里有1200 4个小数
        layers = []
        last_out = 4 # first time
        conv1d_out_channels = 1
        last_input = 21 #lenth
        # is_first = True
        # for k,params in dict_params.items():
        idx = 0
        last_type = ""

        # struct_list = [{'dropout': {'dropout': {'action': 0.16, 'log_prob': -2.341522216796875, 'prob': 0.09618111699819565}}},
        #  {'dropout': {'dropout': {'action': 0.16, 'log_prob': -2.344510555267334, 'prob': 0.09589412063360214}}},
        #  {'linear': {'linear_out': {'action': 470, 'log_prob': -3.8884224891662598, 'prob': 0.020477622747421265}}},
        #  {'activate': {'active': {'action': 'PReLU', 'log_prob': -2.092571496963501, 'prob': 0.1233694776892662}}}, {
        #      'pooling': {'pool_type': {'action': 'max', 'log_prob': -0.6207250356674194, 'prob': 0.5375545620918274},
        #                  'conv_pool': {'action': 4, 'log_prob': -1.205686330795288, 'prob': 0.299486368894577}}}, {
        #      'conv': {
        #          'conv1d_out_channels': {'action': 110, 'log_prob': -4.057315826416016, 'prob': 0.01729538105428219},
        #          'conv1d_kernel_size': {'action': 3, 'log_prob': -1.5888023376464844, 'prob': 0.20416998863220215},
        #          'conv_padding': {'action': 0, 'log_prob': -0.7189777493476868, 'prob': 0.48725008964538574}}},
        #  {'dropout': {'dropout': {'action': 0.1, 'log_prob': -2.474836826324463, 'prob': 0.08417672663927078}}},
        #  {'linear': {'linear_out': {'action': 270, 'log_prob': -3.7905914783477783, 'prob': 0.022582240402698517}}},
        #  {'dropout': {'dropout': {'action': 0.15, 'log_prob': -2.347360610961914, 'prob': 0.09562120586633682}}}, {
        #      'conv': {
        #          'conv1d_out_channels': {'action': 400, 'log_prob': -3.838366746902466, 'prob': 0.021528733894228935},
        #          'conv1d_kernel_size': {'action': 9, 'log_prob': -1.6378591060638428, 'prob': 0.1943957805633545},
        #          'conv_padding': {'action': 1, 'log_prob': -0.6679670810699463, 'prob': 0.5127499103546143}}},
        #  {'batch_norm': {'out': {'action': 0.07, 'log_prob': -2.3047401905059814, 'prob': 0.099784716963768}}},
        #  {'activate': {'active': {'action': 'Sigmoid', 'log_prob': -2.0405666828155518, 'prob': 0.12995505332946777}}}]
        # struct_list = [{'pooling': {'pool_type': {'action': 'avg', 'log_prob': -0.7400567531585693, 'prob': 0.4770868122577667},
        #               'conv_pool': {'action': 2, 'log_prob': -1.1272640228271484, 'prob': 0.32391828298568726}}}, {
        #      'pooling': {'pool_type': {'action': 'avg', 'log_prob': -0.7400162220001221, 'prob': 0.4771061837673187},
        #                  'conv_pool': {'action': 3, 'log_prob': -1.1896824836730957, 'prob': 0.3043178617954254}}}]

        for dict_struct in struct_list:
            for layer_type, dict_params in dict_struct.items():
                # if idx == 0 and layer_type != "conv":
                #     last_out=last_out*last_input
                #     layers.append(nn.Flatten())
                # if idx != 0 and last_type != "conv" and layer_type == "conv":
                #     raise ValueError("last type not conv this type conv")
                if layer_type == "conv":
                    conv1d_out_channels = dict_params.get("conv1d_out_channels").get("action")
                    conv1d_kernel_size = dict_params.get("conv1d_kernel_size").get("action")
                    # conv_batchnorm = dict_params.get("conv_batch_norm").get("action")
                    # conv_active = dict_params.get("conv_active").get("action")
                    # pool_type = dict_params.get("pool_type").get("action")
                    # conv_pool = dict_params.get("conv_pool").get("action")
                    # conv_dropout = dict_params.get("conv_dropout").get("action")
                    padding = dict_params.get("conv_padding").get("action")
                    # last_out  c
                    layer_model = nn.Conv1d(last_out, conv1d_out_channels, conv1d_kernel_size, 1, padding=padding,
                              padding_mode="replicate")
                    layer_model.register_forward_pre_hook(conv_pre_hook)
                    layer_model.register_forward_hook(conv_after_forward)
                    layers.append(layer_model)


                    # if conv_batchnorm is 1:
                    #     layers.append(nn.BatchNorm1d(conv1d_out_channels))
                    # after_conv padding 以后大小一样 在kernel =1 时不一样
                    last_input = (last_input + 2 * padding - conv1d_kernel_size + 1)
                    if last_input < 0:
                        raise ValueError("last_input < 0")
                    # self.__setattr__("conv1d@"+str(layer_no),nn.Conv1d(4, conv1d_out_channels, conv1d_kernel_size, 1))
                    # layers.append(activation_functions[conv_active])
                    # self.__setattr__("conv_active@"+str(layer_no),activation_functions[conv_active])
                    # if pool_type == "avg":
                    #     layers.append(nn.AvgPool1d(conv_pool, 1))
                    # elif pool_type == "max":
                    #     layers.append(nn.MaxPool1d(conv_pool, 1))
                    # else:
                    #     conv_pool = 0

                    # self.__setattr__("conv_pool@"+str(layer_no),nn.AvgPool1d(conv_pool))  # size of window 2  (15,80)
                    # layers.append(nn.Dropout(p=conv_dropout))

                    # self.__setattr__("conv_dropout@" + str(layer_no), nn.Dropout(p=conv_dropout))
                    # layers.append(nn.Flatten())
                    # self.flatten = nn.Flatten()
                    padding = 0
                    # after pooling
                    # if conv_pool is not 0:
                    #     last_input = (last_input + 2 * padding - conv_pool + 1)

                    last_out = conv1d_out_channels
                    # 这个只有再下一个是linear的时候需要
                    # layers.append(nn.Flatten())
                    # last_out = conv1d_out_channels * last_input
                    # if last_out <= 0:
                    #     raise ValueError("input < 0")
                elif layer_type == "linear":
                    linear_out = dict_params.get("linear_out").get("action")
                    # linear_batchnorm = dict_params.get("linear_batch_norm").get("action")
                    # linear_active = dict_params.get("linear_active").get("action")
                    # linear_dropout = dict_params.get("linear_dropout").get("action")

                    layer_model = nn.Linear(last_out, linear_out)
                    layer_model.register_forward_pre_hook(linear_pre_hook)
                    layer_model.register_forward_hook(linear_after_hook)
                    layers.append(layer_model)

                    # if linear_batchnorm is 1:
                    #     layers.append(nn.BatchNorm1d(linear_out))
                    # layers.append(activation_functions[linear_active])
                    # layers.append(nn.Dropout(p=linear_dropout))
                    last_out = linear_out


                elif layer_type == "multiheadattention":
                    # (length, batch size, d_model)
                    # d_model: the number of expected features in the input
                    d_model = last_out#dict_params.get("d_model").get("action")
                    nhead = dict_params.get("nhead").get("action")
                    dropout = dict_params.get("dropout").get("action")
                    layer_model = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                    layer_model.register_forward_pre_hook(mutiheadattention_pre_hook)
                    layer_model.register_forward_hook(mutiheadattention_after_forward)
                    layers.append(layer_model)
                    last_out = d_model
                # elif layer_type is "embedding": #embedding 一定要第一层处理 而且不能用one hot encoding
                #     embedding_size = dict_params.get("embedding_size").get("action")
                #     layer_model = nn.Embedding(last_out, embedding_size)
                #     layer_model.register_forward_pre_hook(embedding_pre_hook)
                #     layers.append(layer_model)
                #     last_out = embedding_size
                elif layer_type is "activate":
                    active = dict_params.get("active").get("action")
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
                        # 'None': nn.Identity()
                    }
                    layers.append(activation_functions[active])
                elif layer_type is "dropout":
                    dropout = dict_params.get("dropout").get("action")
                    layers.append(nn.Dropout(p=dropout))
                elif layer_type is "batch_norm":
                    # batch_norm_out = dict_params.get("out").get("action")
                    layer_model = nn.BatchNorm1d(last_out)
                    layer_model.register_forward_pre_hook(batch_norm_pre_hook)
                    layer_model.register_forward_hook(batch_norm_after_hook)
                    layers.append(layer_model)
                elif layer_type is "pooling":
                    pool_type = dict_params.get("pool_type").get("action")
                    conv_pool = dict_params.get("conv_pool").get("action")
                    padding = 0
                    last_input = (last_input + 2 * padding - conv_pool + 1)
                    if pool_type == "avg":
                        layer_model = nn.AvgPool1d(conv_pool, 1)
                    elif pool_type == "max":
                        layer_model = nn.MaxPool1d(conv_pool, 1)

                    layer_model.register_forward_pre_hook(polling_pre_hook)
                    layer_model.register_forward_hook(polling_after_hook)
                    layers.append(layer_model)
                idx += 1
                last_type = layer_type




        # self.sequential = nn.Sequential(*layers)
        self.sequential = MySequential(*layers)
        if last_type == "conv" or last_type == "pooling":
            last_out = last_out * last_input
        elif last_type == "multiheadattention":
            last_out =d_model * last_input
        else:
            last_out = last_out * last_input
        self.linear_1 = nn.Linear(last_out, 1)  # (None, 1)
        self.linear_1.register_forward_pre_hook(other_pre_hook)



    def forward(self, x):

        x.shape_name = ("batch","hot","len")
        out = self.sequential(x)

        out = self.linear_1(out)

        result = out.squeeze(1)
        return result




# 这里有个问题是state怎么搞
# 我的想法是batch里每个输入的atcg碱基作为state
'''

'''
class PolicyGradientNetwork(nn.Module):

    def __init__(self,architecture_map=None,hidden_size=64,max_layer=22):
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

        self.lstm1 = nn.LSTMCell(state_size, hidden_size)
        self.struct_map = {}
        self.c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)
        # h_t = state
        self.h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)
        self.max_layer = max_layer
        self.type_layer_common = nn.Linear(state_size, 64)
        # 4 layer type 这里应该每一层分开用不同的type比较好
        for i in range(max_layer):
            self.__setattr__("type_layer"+str(i), nn.Linear(64, 8))
        # self.type_layer2 = nn.Linear(32, 7)
        # self.embedding_linear=nn.Linear(state_size, len(self.architecture_map["linear"]["embedding"]))

        for dict_k,dict_v in architecture_map.items():
            for k, v in dict_v.items():
                k_str = k #+ "@" + str(i)
                self.__setattr__(k_str, nn.Linear(state_size, len(v)))
                # linear = nn.Linear(16, len(v))
                # self.struct_map[k] = linear

        # for i in range(self.max_linear_layer):
        # for k, v in self.architecture_map["linear"].items():
        #     k_str = k #+ "@" + str(i)
        #     self.__setattr__(k_str, nn.Linear(state_size, len(v)))
        #         # linear = nn.Linear(16, len(v))
        #         # self.struct_map[k] = linear
        #
        # for k, v in self.architecture_map["embedding"].items():
        #     # k_str = k + "@" + str(i)
        #     self.__setattr__(k, nn.Linear(state_size, len(v)))
        #
        # for k, v in self.architecture_map["multiheadattention"].items():
        #     # k_str = k + "@" + str(i)
        #     self.__setattr__(k, nn.Linear(state_size, len(v)))



    def forward(self, state,conv_layer=2,linear_layer=2):
        # flt = self.flatten(state)
        # action_prob_list = []
        # hid1 = torch.tanh(self.fc1(flt))
        # hid2 = torch.tanh(self.fc2(hid1))
        total_log_prob = 0
        element_count = 0
        list_struct = []


        # action_prob_map["conv"] = {}
        # action_prob_map["linear"] = {}
        c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)
        h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)
        list_type_prob = []
        log_prob = 0
        # 用rnn这个先生成type 然后根据type生成对应种类的明细
        for i in range(self.max_layer):
            action_prob_map = {}
            h_t, c_t = self.lstm1(state, (h_t, c_t))
            hid = c_t.squeeze() #
            output = self.type_layer_common(hid)
            type_layer_str = "type_layer"+str(i)
            type_layer_linear = self.__getattr__(type_layer_str)
            output = type_layer_linear(output)
            output = F.softmax(output, dim=-1)
            action_dist = Categorical(output)
            action_index = action_dist.sample()
            type_log_prob = action_dist.log_prob(
                action_index)  # log_prob returns the log of the probability density/mass function evaluated at the given sample value.
            type_prob = torch.exp(type_log_prob)
            list_type_prob.append((action_index.item(),type_prob.item()))


            layer_type = action_index.item()
            total_log_prob += type_log_prob
            # 0 conv 1 linear 2 "embedding" 3 end 4 end 5 activate function 6 batch norm 7 dropout 8 LayerNorm
            if layer_type is 0:
                action_prob_map["conv"] = {}
                for k, v in self.architecture_map["conv"].items():
                    h_t, c_t = self.lstm1(state, (h_t, c_t))
                    k_str = k #+ "@" + str(i)
                    linear_func = self.__getattr__(k_str)
                    hid = self.h_t.squeeze()
                    result = linear_func(hid)
                    result_softmax = F.softmax(result, dim=-1)
                    # action_prob = torch.sum(result_softmax, dim=0)
                    action_index,log_prob,prob = self.sample(result_softmax)
                    action = v[action_index]

                    action_prob_map["conv"][k_str] = {}
                    action_prob_map["conv"][k_str]["action"] = action
                    action_prob_map["conv"][k_str]["log_prob"] = log_prob.item()
                    action_prob_map["conv"][k_str]["prob"] = prob.item()
                    element_count += 1
                    total_log_prob += log_prob
            elif layer_type is 1:
                action_prob_map["linear"] = {}
                for k, v in self.architecture_map["linear"].items():
                    h_t, c_t = self.lstm1(state, (h_t, c_t))
                    k_str = k #+ "@" + str(i)
                    linear_func = self.__getattr__(k_str)
                    hid = h_t.squeeze()
                    result = linear_func(hid)
                    result_softmax = F.softmax(result, dim=-1)
                    action_index, log_prob, prob = self.sample(result_softmax)
                    action = v[action_index]

                    action_prob_map["linear"][k_str] = {}
                    action_prob_map["linear"][k_str]["action"] = action
                    action_prob_map["linear"][k_str]["log_prob"] = log_prob.item()
                    action_prob_map["linear"][k_str]["prob"] = prob.item()
                    element_count += 1
                    total_log_prob += log_prob
            elif layer_type is 2: #"multiheadattention"
                action_prob_map["multiheadattention"] = {}
                for k, v in self.architecture_map["multiheadattention"].items():
                    h_t, c_t = self.lstm1(state, (h_t, c_t))
                    k_str = k #+ "@" + str(i)

                    hid = h_t.squeeze()
                    linear_func = self.__getattr__(k_str)

                    result = linear_func(hid)
                    result_softmax = F.softmax(result, dim=-1)
                    action_index, log_prob, prob = self.sample(result_softmax)
                    action = v[action_index]

                    action_prob_map["multiheadattention"][k_str] = {}
                    action_prob_map["multiheadattention"][k_str]["action"] = action
                    action_prob_map["multiheadattention"][k_str]["log_prob"] = log_prob.item()
                    action_prob_map["multiheadattention"][k_str]["prob"] = prob.item()
                    element_count += 1
                    total_log_prob += log_prob
            elif layer_type is 3:
                # element_count += 1
                # total_log_prob += torch.tensor(-3.0)

                break
            elif layer_type is 4:
                action_prob_map["activate"] = {}
                for k, v in self.architecture_map["activate"].items():
                    h_t, c_t = self.lstm1(state, (h_t, c_t))
                    k_str = k #+ "@" + str(i)

                    hid = h_t.squeeze()
                    linear_func = self.__getattr__(k_str)

                    result = linear_func(hid)
                    result_softmax = F.softmax(result, dim=-1)
                    action_index, log_prob, prob = self.sample(result_softmax)
                    action = v[action_index]

                    action_prob_map["activate"][k_str] = {}
                    action_prob_map["activate"][k_str]["action"] = action
                    action_prob_map["activate"][k_str]["log_prob"] = log_prob.item()
                    action_prob_map["activate"][k_str]["prob"] = prob.item()
                    element_count += 1
                    total_log_prob += log_prob
            elif layer_type is 5:
                action_prob_map["dropout"] = {}
                for k, v in self.architecture_map["dropout"].items():
                    h_t, c_t = self.lstm1(state, (h_t, c_t))
                    k_str = k  # + "@" + str(i)

                    hid = h_t.squeeze()
                    linear_func = self.__getattr__(k_str)

                    result = linear_func(hid)
                    result_softmax = F.softmax(result, dim=-1)
                    action_index, log_prob, prob = self.sample(result_softmax)
                    action = v[action_index]

                    action_prob_map["dropout"][k_str] = {}
                    action_prob_map["dropout"][k_str]["action"] = action
                    action_prob_map["dropout"][k_str]["log_prob"] = log_prob.item()
                    action_prob_map["dropout"][k_str]["prob"] = prob.item()
                    element_count += 1
                    total_log_prob += log_prob
            elif layer_type is 6:
                action_prob_map["batch_norm"] = {}
                for k, v in self.architecture_map["batch_norm"].items():
                    h_t, c_t = self.lstm1(state, (h_t, c_t))
                    k_str = k  # + "@" + str(i)

                    hid = h_t.squeeze()
                    linear_func = self.__getattr__(k_str)

                    result = linear_func(hid)
                    result_softmax = F.softmax(result, dim=-1)
                    action_index, log_prob, prob = self.sample(result_softmax)
                    action = v[action_index]

                    action_prob_map["batch_norm"][k_str] = {}
                    action_prob_map["batch_norm"][k_str]["action"] = action
                    action_prob_map["batch_norm"][k_str]["log_prob"] = log_prob.item()
                    action_prob_map["batch_norm"][k_str]["prob"] = prob.item()
                    element_count += 1
                    total_log_prob += log_prob
            elif layer_type is 7:
                action_prob_map["pooling"] = {}
                for k, v in self.architecture_map["pooling"].items():
                    h_t, c_t = self.lstm1(state, (h_t, c_t))
                    k_str = k  # + "@" + str(i)

                    hid = h_t.squeeze()
                    linear_func = self.__getattr__(k_str)

                    result = linear_func(hid)
                    result_softmax = F.softmax(result, dim=-1)
                    action_index, log_prob, prob = self.sample(result_softmax)
                    action = v[action_index]

                    action_prob_map["pooling"][k_str] = {}
                    action_prob_map["pooling"][k_str]["action"] = action
                    action_prob_map["pooling"][k_str]["log_prob"] = log_prob.item()
                    action_prob_map["pooling"][k_str]["prob"] = prob.item()
                    element_count += 1
                    total_log_prob += log_prob
            # elif layer_type is 2: #"embedding" # 之后单独处理
            #     action_prob_map["embedding"] = {}
            #     for k, v in self.architecture_map["embedding"].items():
            #         h_t, c_t = self.lstm1(state, (h_t, c_t))
            #         k_str = k #+ "@" + str(i)
            #
            #         hid = h_t.squeeze()
            #         linear_func = self.__getattr__(k_str)
            #
            #         result = linear_func(hid)
            #         result_softmax = F.softmax(result, dim=-1)
            #         action_index, log_prob, prob = self.sample(result_softmax)
            #         action = v[action_index]
            #
            #         action_prob_map["embedding"][k_str] = {}
            #         action_prob_map["embedding"][k_str]["action"] = action
            #         action_prob_map["embedding"][k_str]["log_prob"] = log_prob
            #         action_prob_map["embedding"][k_str]["prob"] = prob
            #         element_count += 1
            #         total_log_prob += log_prob
            else:
                print("unknow type",layer_type)
                continue
            list_struct.append(action_prob_map)
        logging.info("type index  type prob" + str(list_type_prob))


        return list_struct,h_t,total_log_prob,element_count

    def sample(self,item_action_prob):
        action_dist = Categorical(item_action_prob)
        action_index = action_dist.sample()  # .unsqueeze(1)  这里就是根据概率进行采样
        log_prob = action_dist.log_prob(
            action_index)  # log_prob returns the log of the probability density/mass function evaluated at the given sample value.
        prob = torch.exp(log_prob)
        return action_index,log_prob,prob


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
        self.activation_functions_list = ['Sigmoid','Tanh','ReLU','LeakyReLU','ELU','Hardswish','ReLU6','PReLU']
        self.pool_type_list = ['avg','max']
        self.pool_kernel_list = [2,3,4]
        self.padding_list = [0,1]
        self.conv1d_out_channels_list = [i for i in range(521) if i>9 and i%10==0] #[80,90,100,110]
        self.conv1d_kernel_size_list = [1,3,5,7,9]  #[7,5,1]
        self.drop_out_list = [0.02,0.05,0.1,0.15,0.2,0.07,0.12,0.16,0.23,0.25,0.3]
        self.conv_num_list = [1,2,3,4]
        self.linear_num_list = [1,2,3,4]
        self.conv_batch_norm_list = [0,1]
        self.batch_norm_list = [0, 1]

        self.linear40_40_out_features_list = [i for i in range(500) if i>2 and i%10==0]#[10,20,40,80]
        self.need_pool = [0,1]

        self.architecture_map ={
            "conv":{
                "conv1d_out_channels": self.conv1d_out_channels_list,
                "conv1d_kernel_size": self.conv1d_kernel_size_list,
                # "conv_batch_norm":self.conv_batch_norm_list,
                # "conv_active": self.activation_functions_list,
                "conv_padding": self.padding_list,
                # "conv_pool": self.pool_kernel_list,
                # "pool_type":self.pool_type_list,

            },
            # "conv_num":self.conv_num_list,
            "linear":{
                "linear_out": self.linear40_40_out_features_list,
                # "linear_batch_norm": self.batch_norm_list,
                # "linear_active": self.activation_functions_list,
                # "linear_dropout": self.drop_out_list
            },
            # "linear_num": self.linear_num_list,
            "embedding":{
                #"vocab_size": [4], 因为只有ATCG所以这里定死了只能是4
                "embedding_size": [i for i in range(100) if i>2 and i%5==0] #这个其实就相当于输出了
            },
            "multiheadattention":{
                "d_model": [i for i in range(100) if i>2 and i%2==0],
                "nhead":[1, 2, 3, 4, 5, 6, 7, 8, 9],#因为dmodel% nhead要等于0
                "dropout":self.drop_out_list
            },
            "<end>":{},
            "activate":{
                "active":self.activation_functions_list
            },
            "dropout":{
                "dropout":self.drop_out_list
            },
            "pooling":{
                "pool_type":self.pool_type_list,
                "conv_pool": self.pool_kernel_list,
            },
            "batch_norm":{
                "out":self.drop_out_list
            },
        }

        self.action_list = []

        self.network = PolicyGradientNetwork(self.architecture_map).to(device)
        self.optimizer = optim.SGD(self.network.parameters(),lr=0.00001)
        self.try_load_checkpoint()


    def set_new_num(self,new_conv_num,new_linear_num):
        self.conv_num = new_conv_num
        self.linear_num = new_linear_num

    # 儲存及載入模型參數
    def checkpoint_save(self, loss, epoch,reward):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            "reward":reward
        }, PATH)



    def try_load_checkpoint(self):
        checkpath = Path(PATH)
        if checkpath.exists():
            checkpoint = torch.load(PATH)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        else:
            logging.info(f"no checkpoints found at {checkpath}!")



    def learn(self, rewards,log_prob):
        # 损失函数要是一个式子
        loss = -torch.mean(log_prob)*rewards
        logging.info("reinfor loss "+str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.checkpoint_save(loss, 1, rewards)


    def learn1(self, rewards,log_prob):
        # 损失函数要是一个式子
        x = -torch.mean(log_prob)*rewards/1000
        loss = pow(0.5,x)
        logging.info("log_prob:"+str(log_prob)+" x : "+str(x)+" reinfor loss "+str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.checkpoint_save(loss,1,rewards)
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


    def sample_action(self,action_prob_map_list,regression_params,key=""):
        for action_prob_map in action_prob_map_list:
            for k,item_action_prob in action_prob_map.items():
                for key,value in item_action_prob:
                    action_dist = Categorical(item_action_prob)
                    action_index = action_dist.sample() #.unsqueeze(1)  这里就是根据概率进行采样
                    log_prob = action_dist.log_prob(action_index)  #log_prob returns the log of the probability density/mass function evaluated at the given sample value.
                    prob = torch.exp(log_prob)

                action = item_action_prob[action_index.item()]
                regression_params[k] = {}
                regression_params[k]["action"] = action
                self.total_log_prob = self.total_log_prob + log_prob
                regression_params[k]["log_prob"] = log_prob.item()
                regression_params[k]["prob"] = prob.item()





    def sample(self, state,conv_num=2,linear_num=2):

        action_prob_map,new_state,total_log_prob,element_count = self.network(state,self.conv_num,self.linear_num) #torch.cuda.FloatTensor(
        # action_prob = torch.sum(item_action_prob,dim=0)   # To sum over all rows (i.e. for each column)  size = [1, ncol]
        if len(action_prob_map) is 0:
            return None,total_log_prob,None
        regression_params = {}
        self.total_log_prob = total_log_prob
        # self.sample_action(action_prob_map,regression_params)

        #
        avg_log_prob = self.total_log_prob/element_count


        # regression_params = {
        #     "conv1d_out_channels":action[0],
        #     "conv1d_kernel_size":action[1],
        #     "linear1200_80_out_features":action[2],
        #     "linear80_40_out_features":action[3],
        #     "linear40_40_out_features":action[4],
        #
        # }


        logging.info(action_prob_map)
        return action_prob_map,avg_log_prob,new_state





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

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # optimizer 使用 Adam

        num_epoch = 40

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



    agent = PolicyGradientAgent()

    agent.network.train()  # 訓練前，先確保 network 處在 training 模式

    NUM_BATCH = 5860000  # 總共更新 7600 次

    hidden_size = 64
    state = torch.zeros(1, hidden_size, dtype=torch.float, device=device)

    for batch in range(NUM_BATCH):
        # train_set, val_set = torch.utils.data.random_split(all_train_set, [12000, 2999])

        # task = Task(wt_train_set, wt_test_set)
        task = Task(eSpCas_train_set, eSpCas_test_set)
        # 暂时先和数据分离开 state和数据无关
        # state = task.get_state()

        actionparam,log_prob,newstate = agent.sample(state)
        if actionparam is None:
            logging.info("len 1   -1000")
            reward = -1000
            # log_prob = torch.tensor(-3.0) #不能这里设置
            agent.learn(reward, log_prob)
            continue

        struct_len = len(actionparam)
        # new_conv_num = actionparam["conv_num"]
        # new_linear_num = actionparam["linear_num"]
        try:
            model = Regression(actionparam).to(device)
            logging.info(model)
            # agent.set_new_num(new_conv_num.get("action"),new_linear_num.get("action"))
            tr_load = DataLoader(eSpCas_train_set, batch_size=512, shuffle=False)
            val_load = DataLoader(eSpCas_test_set, batch_size=512, shuffle=False)
            action_loss = task.train(model,tr_load)
            evaluate_loss,df = task.evaluate(model,val_load)
            rho, p = spearmanr(df["predict"], df["ground truth"])

            if math.isnan(rho):
                rho = 0
                p = 0

            #  以前main函数训练的结果记为baseline  reward 基于 baseline 来
            mean_loss = action_loss*0.15+evaluate_loss*0.85
            spearman_reward = rho * 1000
            struct_factor = 700/struct_len
            base_line = 650
            reward = spearman_reward - struct_factor-base_line
            global max_reward
            global max_spearman

            if rho>max_spearman:
                max_spearman=rho
                logging.error("max_spearman:" + str(max_spearman) + " architecture:" + str(actionparam))

            if reward < -1000 or math.isnan(reward):
                logging.error("bad architecture "+str(reward)+"fix to -10000")
                reward = -1000


            if reward > max_reward:
                max_reward = reward
                logging.error("max_reward:"+ str(max_reward) +" architecture:" + str(actionparam))
                if max_reward >0 :
                    reward = reward * 3
                    logging.info("new reward:***********************3333")

            if reward > 0:
                # reward = reward * 1.2
                logging.info("reward:***********************1.2")
                logging.info("reward:"+str(reward)+"mean_loss:"+str(mean_loss)+" action_loss"+str(action_loss)+"evaluate_loss"+str(evaluate_loss)+"spearmanr "+str(rho) + " p "+str(p))

            logging.info("reward:"+str(reward)+"struct_factor:"+str(struct_factor)+" spearman_reward"+str(spearman_reward)+"evaluate_loss"+str(evaluate_loss)+"spearmanr "+str(rho) + " p "+str(p))
            agent.learn(reward,log_prob)
        except ValueError as vex:
            logging.info("!!!!!!!!!!"+str(vex)+"!!!!!!!!!!!!!!!!!!!!!!!!!!!!input <0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            reward = -1000
            agent.learn(reward, log_prob)
        except Exception as ex:
            logging.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Except"+str(ex))
            reward = -1000
            if log_prob is None:
                log_prob = torch.tensor([5.0],requires_grad = True)
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

