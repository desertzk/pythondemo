
import numpy as np

import torch
import torch.nn as nn

import pandas as pd
from numpy import zeros

from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.distributions import Categorical
from os import path
import math
import json
import torch.nn.functional as F
import logging
from scipy.stats import spearmanr,pearsonr
from itertools import islice
import operator
from collections import OrderedDict
from torch._jit_internal import _copy_to_script_wrapper
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Any, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt

T = TypeVar('T')

g_batch_size = 512

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("hf1debugpadding211106.log"),
        logging.StreamHandler()
    ]
)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
logging.info(device)
# cudnn.benchmark = True
PATH = "hf1_trans_skip_checkpoint_last_un.pt"

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


def data_load_cas():
    train_data = pd.read_csv('_data_/SpCas9-HF1.csv')

    bp34_col = train_data["Input_Sequence"]
    xcas_efficiency = train_data["Indel_Norm"]
    # classification = train_data["classification"]

    return bp34_col,xcas_efficiency


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

hf1bp,hf1_efficiency = data_load_cas()

# wt_train_x = wt_df_clean["21mer"]
# wt_efficiency = wt_df_clean["Wt_Efficiency"]
spcas9bp_train_x = PREPROCESS_ONE_HOT(hf1bp,23)
hf1bp_train_x_for_torch = np.transpose(spcas9bp_train_x,(0,2,1))
# test_x__for_torch = np.transpose(test_x,(0,2,1))
cas_efficiency_set = RNADataset(hf1bp_train_x_for_torch,hf1_efficiency)



HF1_train_set,HF1_test_set = torch.utils.data.random_split(cas_efficiency_set, [48354, 8534])



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
    c_idx = input[0].shape_name.index("hot")
    b_idx = input[0].shape_name.index("batch")
    l_idx = input[0].shape_name.index("len")
    change_input = input[0].permute(b_idx, l_idx, c_idx)


    # if input[0].shape_name[1] == "batch":  #前面有mutiheadattion 交换了 batch
    #     change_input = input[0].permute(1, 0, 2)
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


def active_after_hook(model, input,result):
    # 如果是前面有conv的需要换位指定 c
    # (N,C,L)
    if input[0].ndim == 3:
        result.shape_name = ("batch","hot", "len")
    return result

def active_pre_hook(model, input):
    # 如果是前面有conv的需要换位指定 c
    # (N,C,L)
    c_idx = input[0].shape_name.index("hot")
    b_idx = input[0].shape_name.index("batch")
    l_idx = input[0].shape_name.index("len")
    change_input = input[0].permute(b_idx, c_idx, l_idx)
    return change_input

def dropout_after_hook(model, input,result):
    # 如果是前面有conv的需要换位指定 c
    # (N,C,L)
    if input[0].ndim == 3:
        result.shape_name = ("batch","hot", "len")
    return result

def dropout_pre_hook(model, input):
    # 如果是前面有conv的需要换位指定 c
    # (N,C,L)
    c_idx = input[0].shape_name.index("hot")
    b_idx = input[0].shape_name.index("batch")
    l_idx = input[0].shape_name.index("len")
    change_input = input[0].permute(b_idx, c_idx, l_idx)
    return change_input

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
        for key, module in enumerate(arg):
            self.add_module(key, module)

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

    @_copy_to_script_wrapper
    def __iter__(self): #-> Iterator[Module]:
        return iter(self._modules.items())

    # def forward(self, input):
    #     for k,module in self:
    #         # print(input.shape)
    #         input1 = module(input)
    #         if not hasattr(input1,"shape_name"):
    #             input1.shape_name = input.shape_name
    #         input = input1
    #     return input1

    def change_model_input(self,channel,module):
        if type(module) is nn.Conv1d:
            if module.in_channels != channel:
                module.in_channels = channel
            # module = nn.Conv1d(channel,module.out_channels,module.kernel_size,module.stride,module.padding,padding_mode=module.padding_mode).to(device)   不能每次都新建 要不然里面的参数都没经过训练了
            # module.register_forward_pre_hook(conv_pre_hook)
            # module.register_forward_hook(conv_after_forward)
        elif type(module) is nn.MultiheadAttention:
            if module.embed_dim != channel:
                module.embed_dim = channel
            # module = nn.MultiheadAttention(channel,module.num_heads, dropout=module.dropout).to(device)
            # module.register_forward_pre_hook(mutiheadattention_pre_hook)
            # module.register_forward_hook(mutiheadattention_after_forward)
        elif type(module) is nn.Linear:
            if module.in_features!=channel:
                pad_weight = torch.nn.functional.pad(module.weight,(0,abs(channel-module.in_features)))
                pad_bias = torch.nn.functional.pad(module.bias,(0,abs(channel-module.in_features)))
                module.weight = nn.Parameter(pad_weight)
                module.bias = nn.Parameter(pad_bias)
                module.in_features = channel
            # module = nn.Linear(channel,module.out_features).to(device)
            # module.register_forward_pre_hook(linear_pre_hook)
            # module.register_forward_hook(linear_after_hook)
        elif type(module) is nn.BatchNorm1d:
            if module.num_features != channel:
                pad_weight = torch.nn.functional.pad(module.weight,(0,abs(channel-module.num_features)),value=1.0)
                pad_bias = torch.nn.functional.pad(module.bias,(0,abs(channel-module.num_features)))
                module.weight = nn.Parameter(pad_weight)
                module.bias = nn.Parameter(pad_bias)
                module.num_features = channel

            # module = nn.BatchNorm1d(channel).to(device)
            # module.register_forward_pre_hook(batch_norm_pre_hook)
            # module.register_forward_hook(batch_norm_after_hook)
        # elif type(module) is nn.Conv1d:
        # elif type(module) is nn.Conv1d:

        return module


    def forward(self, input):
        tmp_list = []
        skip_list = []

        # old_layer_num = -1
        for k,module in self:
            # print(input.shape,input.shape_name)
            # if input.shape[0] ==159:
            #     print("rerw")
            # 这里的model要根据input适应一下
            # channel_idx = input.shape_name.index("hot")
            # channel = input.shape[channel_idx]
            # module = self.change_model_input(channel,module)
            pre_skip_no = -1
            layer_no, operator_type, op_value = k.split('|')
            list_len = len(tmp_list)

            if len(skip_list) != 0:
                pre_skip_dict = skip_list[0]
                pre_skip_no = pre_skip_dict["skip_no"]
                pre_skip_input = pre_skip_dict["input"]
                # eliminate duplicate items
                if pre_skip_no < int(layer_no):
                    skip_list.pop(0)

                if pre_skip_no == int(layer_no): #有两个skip到同一层的话 shape 大小有可能会出错
                    skip_list.pop(0)
                    # pre_skip_input+input
                    c_skip_idx = pre_skip_input.shape_name.index("hot")
                    b_skip_idx = pre_skip_input.shape_name.index("batch")
                    l_skip_idx = pre_skip_input.shape_name.index("len")
                    c_idx = input.shape_name.index("hot")
                    b_idx = input.shape_name.index("batch")
                    l_idx = input.shape_name.index("len")

                    firstret1 = pre_skip_input.permute(b_skip_idx, l_skip_idx, c_skip_idx)
                    firstret2 = input.permute(b_idx, l_idx, c_idx)
                    # 一般batch是相同的
                    if firstret1.shape[1] > firstret2.shape[1]:
                        firstret2 = torch.nn.functional.pad(firstret2,
                                                            (0,0,0,firstret1.shape[1]-firstret2.shape[1],0,0))
                    else:
                        firstret1 = torch.nn.functional.pad(firstret1, (
                        0, 0, 0, firstret2.shape[1] - firstret1.shape[1], 0, 0))


                    if firstret1.shape[2] > firstret2.shape[2]:
                        firstret2 = torch.nn.functional.pad(firstret2,
                                                            (0,firstret1.shape[2]-firstret2.shape[2],0,0,0,0))
                    else:
                        firstret1 = torch.nn.functional.pad(firstret1, (
                        0, firstret2.shape[2] - firstret1.shape[2], 0, 0, 0, 0))


                    input = firstret1 + firstret2
                    input.shape_name = ("batch", "len", "hot")


            if operator_type == "add":
                if list_len == 0:
                    tmp_list.append(module)
                    continue
                elif list_len == 1:
                    first = tmp_list.pop(0)
                    firstret = first(input)
                    moduleret = module(input)
                    # first_ret_1dim = torch.flatten(firstret)
                    # module_ret_1dim = torch.flatten(moduleret)
                    # first_ret_1dim_size = len(first_ret_1dim)
                    # module_ret_1dim_size = len(module_ret_1dim)
                    # if first_ret_1dim_size > module_ret_1dim_size:
                    #     module_ret_1dim_new = torch.nn.functional.pad(module_ret_1dim, (0, first_ret_1dim_size-module_ret_1dim_size))
                    #     result_1dim = first_ret_1dim + module_ret_1dim_new
                    #     result = result_1dim.reshape(firstret.shape)
                    #     result.shape_name = firstret.shape_name
                    # else:
                    #     first_ret_1dim_new = torch.nn.functional.pad(first_ret_1dim,
                    #                                               (0, module_ret_1dim_size-first_ret_1dim_size))
                    #     result_1dim = first_ret_1dim_new + module_ret_1dim
                    #     result = result_1dim.reshape(moduleret.shape)
                    #     result.shape_name = moduleret.shape_name

                    c_skip_idx = firstret.shape_name.index("hot")
                    b_skip_idx = firstret.shape_name.index("batch")
                    l_skip_idx = firstret.shape_name.index("len")
                    c_idx = moduleret.shape_name.index("hot")
                    b_idx = moduleret.shape_name.index("batch")
                    l_idx = moduleret.shape_name.index("len")


                    firstret1 = firstret.permute(b_skip_idx, l_skip_idx, c_skip_idx)
                    firstret2 = moduleret.permute(b_idx, l_idx, c_idx)
                    # 一般batch是相同的
                    if firstret1.shape[1] > firstret2.shape[1]:
                        firstret2 = torch.nn.functional.pad(firstret2,
                                                            (0,0,0,firstret1.shape[1]-firstret2.shape[1],0,0))
                    else:
                        firstret1 = torch.nn.functional.pad(firstret1, (
                        0, 0, 0, firstret2.shape[1] - firstret1.shape[1], 0, 0))


                    if firstret1.shape[2] > firstret2.shape[2]:
                        firstret2 = torch.nn.functional.pad(firstret2,
                                                            (0,firstret1.shape[2]-firstret2.shape[2],0,0,0,0))
                    else:
                        firstret1 = torch.nn.functional.pad(firstret1, (
                        0, firstret2.shape[2] - firstret1.shape[2], 0, 0, 0, 0))


                    input1 = firstret1 + firstret2
                    input1.shape_name = ("batch", "len", "hot")



                    # input1 = result
            elif operator_type == "subtract":
                if list_len == 0:
                    tmp_list.append(module)
                    continue
                elif list_len == 1:
                    first = tmp_list.pop(0)
                    firstret = first(input)
                    moduleret = module(input)
                    # first_ret_1dim = torch.flatten(firstret)
                    # module_ret_1dim = torch.flatten(moduleret)
                    # first_ret_1dim_size = len(first_ret_1dim)
                    # module_ret_1dim_size = len(module_ret_1dim)
                    # if first_ret_1dim_size > module_ret_1dim_size:
                    #     module_ret_1dim_new = torch.nn.functional.pad(module_ret_1dim,
                    #                                                   (0, first_ret_1dim_size - module_ret_1dim_size))
                    #     result_1dim = first_ret_1dim - module_ret_1dim_new
                    #     result = result_1dim.reshape(firstret.shape)
                    #     result.shape_name = firstret.shape_name
                    # else:
                    #     first_ret_1dim_new = torch.nn.functional.pad(first_ret_1dim,
                    #                                                  (0, module_ret_1dim_size - first_ret_1dim_size))
                    #     result_1dim = first_ret_1dim_new - module_ret_1dim
                    #     result = result_1dim.reshape(moduleret.shape)
                    #     result.shape_name = moduleret.shape_name
                    c_skip_idx = firstret.shape_name.index("hot")
                    b_skip_idx = firstret.shape_name.index("batch")
                    l_skip_idx = firstret.shape_name.index("len")
                    c_idx = moduleret.shape_name.index("hot")
                    b_idx = moduleret.shape_name.index("batch")
                    l_idx = moduleret.shape_name.index("len")


                    firstret1 = firstret.permute(b_skip_idx, l_skip_idx, c_skip_idx)
                    firstret2 = moduleret.permute(b_idx, l_idx, c_idx)
                    # 一般batch是相同的
                    if firstret1.shape[1] > firstret2.shape[1]:
                        firstret2 = torch.nn.functional.pad(firstret2,
                                                            (0,0,0,firstret1.shape[1]-firstret2.shape[1],0,0))
                    else:
                        firstret1 = torch.nn.functional.pad(firstret1, (
                        0, 0, 0, firstret2.shape[1] - firstret1.shape[1], 0, 0))


                    if firstret1.shape[2] > firstret2.shape[2]:
                        firstret2 = torch.nn.functional.pad(firstret2,
                                                            (0,firstret1.shape[2]-firstret2.shape[2],0,0,0,0))
                    else:
                        firstret1 = torch.nn.functional.pad(firstret1, (
                        0, firstret2.shape[2] - firstret1.shape[2], 0, 0, 0, 0))


                    input1 = firstret1 - firstret2
                    input1.shape_name = ("batch", "len", "hot")
                    # input1 = result
            elif operator_type == "multiply":
                if list_len == 0:
                    tmp_list.append(module)
                    continue
                elif list_len == 1:
                    first = tmp_list.pop(0)
                    firstret = first(input)
                    moduleret = module(input)
                    # first_ret_1dim = torch.flatten(firstret)
                    # module_ret_1dim = torch.flatten(moduleret)
                    # first_ret_1dim_size = len(first_ret_1dim)
                    # module_ret_1dim_size = len(module_ret_1dim)
                    # if first_ret_1dim_size > module_ret_1dim_size:
                    #     module_ret_1dim_new = torch.nn.functional.pad(module_ret_1dim,
                    #                                                   (0, first_ret_1dim_size - module_ret_1dim_size))
                    #     result_1dim = first_ret_1dim - module_ret_1dim_new
                    #     result = result_1dim.reshape(firstret.shape)
                    #     result.shape_name = firstret.shape_name
                    # else:
                    #     first_ret_1dim_new = torch.nn.functional.pad(first_ret_1dim,
                    #                                                  (0, module_ret_1dim_size - first_ret_1dim_size))
                    #     result_1dim = first_ret_1dim_new * module_ret_1dim
                    #     result = result_1dim.reshape(moduleret.shape)
                    #     result.shape_name = moduleret.shape_name

                    c_skip_idx = firstret.shape_name.index("hot")
                    b_skip_idx = firstret.shape_name.index("batch")
                    l_skip_idx = firstret.shape_name.index("len")
                    c_idx = moduleret.shape_name.index("hot")
                    b_idx = moduleret.shape_name.index("batch")
                    l_idx = moduleret.shape_name.index("len")


                    firstret1 = firstret.permute(b_skip_idx, l_skip_idx, c_skip_idx)
                    firstret2 = moduleret.permute(b_idx, l_idx, c_idx)
                    # 一般batch是相同的
                    if firstret1.shape[1] > firstret2.shape[1]:
                        firstret2 = torch.nn.functional.pad(firstret2,
                                                            (0,0,0,firstret1.shape[1]-firstret2.shape[1],0,0))
                    else:
                        firstret1 = torch.nn.functional.pad(firstret1, (
                        0, 0, 0, firstret2.shape[1] - firstret1.shape[1], 0, 0))


                    if firstret1.shape[2] > firstret2.shape[2]:
                        firstret2 = torch.nn.functional.pad(firstret2,
                                                            (0,firstret1.shape[2]-firstret2.shape[2],0,0,0,0))
                    else:
                        firstret1 = torch.nn.functional.pad(firstret1, (
                        0, firstret2.shape[2] - firstret1.shape[2], 0, 0, 0, 0))


                    input1 = firstret1 * firstret2
                    input1.shape_name = ("batch", "len", "hot")


            elif operator_type == "concat":#遇到池化层 mat会不一样要特殊处理
                if list_len == 0:
                    tmp_list.append(module)
                    continue
                elif list_len == 1:
                    first = tmp_list.pop(0)


                    firstret = first(input)
                    moduleret = module(input)
                    # if first ==
                    # need permute to same (batch hot len) then  cat
                    c_idx = firstret.shape_name.index("hot")
                    b_idx = firstret.shape_name.index("batch")
                    l_idx = firstret.shape_name.index("len")
                    firstret2 = firstret.permute(b_idx, l_idx, c_idx)
                    c_idxm = moduleret.shape_name.index("hot")
                    b_idxm = moduleret.shape_name.index("batch")
                    l_idxm = moduleret.shape_name.index("len")
                    moduleret2 = moduleret.permute(b_idxm, l_idxm, c_idxm)
                    l_first = firstret2.shape[1]
                    l_module = moduleret2.shape[1]
                    if l_first > l_module:
                        remain = (l_first - l_module)%2
                        pad_size = int((l_first - l_module)/2)
                        moduleret2 = torch.nn.functional.pad(moduleret2,(0,0,pad_size,pad_size+remain),"constant",0)
                    elif l_module > l_first:
                        remain = (l_first - l_module) % 2
                        pad_size = int((l_module - l_first)/2)
                        firstret2 = torch.nn.functional.pad(firstret2,(0,0,pad_size,pad_size+remain),"constant",0)
                    input1 = torch.cat((firstret2,moduleret2),2) # should concat 传0应该是扩充第一个维度
                    input1.shape_name = ("batch","len","hot")
            elif operator_type == "skip":

                skip_no = op_value
                current_no = layer_no
                skip_dict = {}
                skip_dict["skip_no"] = int(skip_no)
                skip_dict["input"] = input

                skip_list.append(skip_dict)
                skip_list.sort(key=lambda s: s["skip_no"])
                input1 = input


            else:
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
        layers = OrderedDict()
        last_out = 4 # first time
        conv1d_out_channels = 1
        last_input = 23 #lenth
        # is_first = True
        # for k,params in dict_params.items():
        idx = 0
        last_type = ""

        #


        out_list = []
        skip_to_layer_list = {}

        # 这里最好写成递归的模式
        for dict_struct in struct_list:
            if idx in skip_to_layer_list.keys():
                try:
                    old_last_out = last_out
                    skip_out = out_list[skip_to_layer_list[idx]-1]
                    last_out = skip_out if skip_out >= old_last_out else old_last_out
                except Exception as ex:
                   print("<<<<<<<<<out_list   Except <<<<<<<<"+str(ex))
            for layer_type, dict_params in dict_struct.items():
                operator_count = 0
                if layer_type == "add":
                    this_layer_out = last_out
                    # for dict_param in dict_params:
                    dict_param = dict_params[0]
                    if dict_param is not {}:
                        operator_count = operator_count + 1
                        layer_model1, last_out1, last_input1, d_model = self.make_layer(dict_param, this_layer_out,
                                                                                      last_input)
                        idxstr = str(idx) + "|add|" + str(operator_count)
                        layers[idxstr] = layer_model1

                    dict_param = dict_params[1]
                    if dict_param is not {}:
                        operator_count = operator_count + 1
                        layer_model1, last_out2, last_input2, d_model = self.make_layer(dict_param, this_layer_out,
                                                                                      last_input)
                        idxstr = str(idx) + "|add|" + str(operator_count)
                        layers[idxstr] = layer_model1

                    mul1 = last_out1 #* last_input1
                    mul2 = last_out2 #* last_input2
                    if mul1 > mul2:#两个里面大的作为下一层的输入
                        last_out = last_out1
                    else:
                        last_out = last_out2

                # 如果上一层是concat那么channel要加
                elif layer_type == "concat":
                    this_layer_out = last_out
                    # for dict_param in dict_params:
                    dict_param = dict_params[0]
                    if dict_param is not {}:
                        operator_count = operator_count + 1
                        layer_model1, last_out1, last_input, d_model = self.make_layer(dict_param, this_layer_out,
                                                                                      last_input)
                        idxstr = str(idx) + "|concat|" + str(operator_count)
                        layers[idxstr] = layer_model1

                    dict_param = dict_params[1]
                    if dict_param is not {}:
                        operator_count = operator_count + 1
                        layer_model1, last_out2, last_input, d_model = self.make_layer(dict_param, this_layer_out,
                                                                                      last_input)
                        idxstr = str(idx) + "|concat|" + str(operator_count)
                        layers[idxstr] = layer_model1
                    # if last_out1>last_out2:#两个里面大的作为下一层的输入
                    #     last_out = last_out1
                    # else:
                    #     last_out = last_out2

                    last_out = last_out1+last_out2
                elif layer_type == "subtract":
                    this_layer_out = last_out
                    # for dict_param in dict_params:
                    dict_param = dict_params[0]
                    if dict_param is not {}:
                        operator_count = operator_count + 1
                        layer_model1, last_out1, last_input1, d_model = self.make_layer(dict_param, this_layer_out,
                                                                                      last_input)
                        idxstr = str(idx) + "|subtract|" + str(operator_count)
                        layers[idxstr] = layer_model1

                    dict_param = dict_params[1]
                    if dict_param is not {}:
                        operator_count = operator_count + 1
                        layer_model1, last_out2, last_input2, d_model = self.make_layer(dict_param, this_layer_out,
                                                                                      last_input)
                        idxstr = str(idx) + "|subtract|" + str(operator_count)
                        layers[idxstr] = layer_model1
                    mul1 = last_out1 #* last_input1
                    mul2 = last_out2 #* last_input2
                    if mul1 > mul2:#两个里面大的作为下一层的输入
                        last_out = last_out1
                    else:
                        last_out = last_out2
                elif layer_type == "multiply":
                    this_layer_out = last_out
                    # for dict_param in dict_params:
                    dict_param = dict_params[0]
                    if dict_param is not {}:
                        operator_count = operator_count + 1
                        layer_model1, last_out1, last_input1, d_model = self.make_layer(dict_param, this_layer_out,
                                                                                      last_input)
                        idxstr = str(idx) + "|multiply|" + str(operator_count)
                        layers[idxstr] = layer_model1

                    dict_param = dict_params[1]
                    if dict_param is not {}:
                        operator_count = operator_count + 1
                        layer_model1, last_out2, last_input2, d_model = self.make_layer(dict_param, this_layer_out,
                                                                                      last_input)
                        idxstr = str(idx) + "|multiply|" + str(operator_count)
                        layers[idxstr] = layer_model1
                    mul1 = last_out1 #* last_input1
                    mul2 = last_out2 #* last_input2
                    if mul1 > mul2:#两个里面大的作为下一层的输入
                        last_out = last_out1
                    else:
                        last_out = last_out2
                elif layer_type == "skip":
                    from_layer = dict_params["from"]
                    to_layer = dict_params["to"]
                    idxstr= str(from_layer)+"|skip|"+str(to_layer)
                    layers[idxstr] = None
                    skip_to_layer_list[to_layer] = from_layer
                else:
                    dict_params = dict_params[0]
                    layer_model1, last_out, last_input,d_model = self.make_layer(dict_params, last_out, last_input)

                    idxstr = str(idx) + "|unary|1"
                    layers[idxstr] = layer_model1

                    #     linear_out = dict_params.get("linear_out").get("action")
                    #     # linear_batchnorm = dict_params.get("linear_batch_norm").get("action")

                idx += 1
                last_type = layer_type
                out_list.append(last_out)

            # self.sequential = nn.Sequential(*layers)
        self.sequential = MySequential(layers)
        if last_type == "conv" or last_type == "pooling":
            last_out = last_out * last_input
        elif last_type == "multiheadattention":
            last_out = d_model * last_input
        else:
            last_out = last_out * last_input
        self.linear_1 = nn.Linear(1, 1)  # (None, 1)
        self.linear_1.register_forward_pre_hook(other_pre_hook)

    def make_layer(self,dict_params,last_out,last_input):
        d_model = 0
        for layer_type, dict_params in dict_params.items():
            if layer_type == "conv":
                conv1d_out_channels = dict_params.get("conv1d_out_channels").get("action")
                conv1d_kernel_size = dict_params.get("conv1d_kernel_size").get("action")
                # conv_batchnorm = dict_params.get("conv_batch_norm").get("action")
                # conv_active = dict_params.get("conv_active").get("action")
                # pool_type = dict_params.get("pool_type").get("action")
                # conv_pool = dict_params.get("conv_pool").get("action")
                # conv_dropout = dict_params.get("conv_dropout").get("action")
                padding = 0#dict_params.get("conv_padding").get("action")
                # last_out  c
                layer_model = nn.Conv1d(last_out, conv1d_out_channels, conv1d_kernel_size, 1, padding=padding,
                                        padding_mode="replicate")
                layer_model.register_forward_pre_hook(conv_pre_hook)
                layer_model.register_forward_hook(conv_after_forward)
                # layers[idxstr](layer_model)

                # if conv_batchnorm is 1:
                #     layers.append(nn.BatchNorm1d(conv1d_out_channels))
                # after_conv padding 以后大小一样 在kernel =1 时不一样
                last_input = (last_input + 2 * padding - conv1d_kernel_size + 1)
                if last_input < 0:
                    raise ValueError("last_input < 0")

                padding = 0
                last_out = conv1d_out_channels


            elif layer_type == "linear":
                linear_out = dict_params.get("linear_out").get("action")
                # linear_batchnorm = dict_params.get("linear_batch_norm").get("action")
                # linear_active = dict_params.get("linear_active").get("action")
                # linear_dropout = dict_params.get("linear_dropout").get("action")

                layer_model = nn.Linear(last_out, linear_out)
                layer_model.register_forward_pre_hook(linear_pre_hook)
                layer_model.register_forward_hook(linear_after_hook)

                last_out = linear_out


            elif layer_type == "multiheadattention":
                # (length, batch size, d_model)
                # d_model: the number of expected features in the input
                d_model = last_out  # dict_params.get("d_model").get("action")
                nhead = dict_params.get("nhead").get("action")
                if d_model % nhead!=0:
                    nhead = 1
                dropout = dict_params.get("dropout").get("action")
                layer_model = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                layer_model.register_forward_pre_hook(mutiheadattention_pre_hook)
                layer_model.register_forward_hook(mutiheadattention_after_forward)

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
                layer_model = activation_functions[active]
                layer_model.register_forward_pre_hook(active_pre_hook)
                layer_model.register_forward_hook(active_after_hook)
            elif layer_type is "dropout":
                dropout = dict_params.get("dropout").get("action")
                layer_model = nn.Dropout(p=dropout)
                layer_model.register_forward_pre_hook(dropout_pre_hook)
                layer_model.register_forward_hook(dropout_after_hook)
            elif layer_type is "batch_norm":
                # batch_norm_out = dict_params.get("out").get("action")
                layer_model = nn.BatchNorm1d(last_out)
                layer_model.register_forward_pre_hook(batch_norm_pre_hook)
                layer_model.register_forward_hook(batch_norm_after_hook)

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
        return layer_model, last_out, last_input,d_model

    def forward(self, x):
        x.shape_name = ("batch","hot","len")
        out = self.sequential(x)
        channel_idx = out.shape_name.index("hot")
        len_idx = out.shape_name.index("len")
        channel = out.shape[channel_idx]
        len = out.shape[len_idx]
        last_out = channel * len
        if last_out!=self.linear_1.in_features:
            self.linear_1 = nn.Linear(last_out, 1).to(device)  # (None, 1)
            self.linear_1.register_forward_pre_hook(other_pre_hook)

        # batch的数值是不能动的 因为后面求loss根据的graoud true 一个batch是512
        out = self.linear_1(out)

        result = out.squeeze(1)
        return result




# 这里有个问题是state怎么搞
# 我的想法是batch里每个输入的atcg碱基作为state
'''

'''
class PolicyGradientNetwork(nn.Module):
    def __init__(self,architecture_map=None,hidden_size=64,max_layer=12):
        super().__init__()
        self.architecture_map = architecture_map

        self.nhid = hidden_size
        # self.hidden = self.init_hidden()
        self.flatten = nn.Flatten()

        # self.linear_num_layer = linear_layer
        # 暂时不用数据来当state
        # state_size = 1
        # for i in state:
        #     state_size = state_size * i
        state_size = hidden_size
        self.total_log_prob = 0
        self.element_count = 0
        # self.lstm1 = nn.LSTMCell(state_size, hidden_size)
        d_model = state_size
        self.transformer = TransformerEncoderLayer(d_model, 2, hidden_size, dropout=0.3)
        self.struct_map = {}
        self.c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)
        # h_t = state
        self.h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)
        self.max_layer = max_layer
        self.type_layer_common = nn.Linear(state_size, 64)
        self.operator_common = nn.Linear(state_size, 32)

        self.activation_functions_list = ['Sigmoid', 'Tanh', 'ReLU', 'LeakyReLU', 'ELU', 'Hardswish', 'ReLU6', 'PReLU']
        self.type_activate = nn.Linear(state_size, len(self.activation_functions_list)) #激活函数的数量

        # self.operator_common = nn.Linear(state_size, 32)
        # 4 layer type 这里应该每一层分开用不同的type比较好
        for i in range(max_layer):
            self.__setattr__("type_layer"+str(i), nn.Linear(64, 8))
            self.__setattr__("operator_count"+str(i), nn.Linear(32, 6))
            self.__setattr__("skip_layer"+str(i), nn.Linear(64, max_layer-i))
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

    # total_log_prob_list to calculate mean
    def unary_operator(self, state,i,total_log_prob_list,hid,list_type_prob):
        # unary operation
        action_prob_map = {}
        # h_t, c_t = self.lstm1(state, (h_t, c_t))
        # hid = c_t.squeeze()  #
        output = self.type_layer_common(hid)
        type_layer_str = "type_layer" + str(i)
        type_layer_linear = self.__getattr__(type_layer_str)
        output = type_layer_linear(output)
        output = F.softmax(output, dim=-1)
        action_dist = Categorical(output)
        action_index = action_dist.sample()
        type_log_prob = action_dist.log_prob(
            action_index)  # log_prob returns the log of the probability density/mass function evaluated at the given sample value.
        type_prob = torch.exp(type_log_prob)
        list_type_prob.append((type_layer_str,action_index.item(), type_prob.item()))

        layer_type = action_index.item()
        self.element_count += 1
        self.total_log_prob += type_log_prob
        # total_log_prob_list.append(type_log_prob)

        # 0 conv 1 linear 2 multiheadattention 3 end 4  activate function 6 batch norm 5 dropout 8 LayerNorm
        if layer_type is 0:
            action_prob_map["conv"] = {}
            for k, v in self.architecture_map["conv"].items():
                # h_t, c_t = self.lstm1(state, (h_t, c_t))
                k_str = k  # + "@" + str(i)
                linear_func = self.__getattr__(k_str)
                # hid = self.h_t.squeeze()
                result = linear_func(hid)
                result_softmax = F.softmax(result, dim=-1)
                # action_prob = torch.sum(result_softmax, dim=0)
                action_index, log_prob, prob = self.sample(result_softmax)
                action = v[action_index]

                action_prob_map["conv"][k_str] = {}
                action_prob_map["conv"][k_str]["action"] = action
                action_prob_map["conv"][k_str]["log_prob"] = log_prob.item()
                action_prob_map["conv"][k_str]["prob"] = prob.item()
                self.element_count += 1
                self.total_log_prob += log_prob
                # total_log_prob_list.append(log_prob)
        elif layer_type is 1:
            action_prob_map["linear"] = {}
            for k, v in self.architecture_map["linear"].items():
                # h_t, c_t = self.lstm1(state, (h_t, c_t))
                k_str = k  # + "@" + str(i)
                linear_func = self.__getattr__(k_str)
                # hid = h_t.squeeze()
                result = linear_func(hid)
                result_softmax = F.softmax(result, dim=-1)
                action_index, log_prob, prob = self.sample(result_softmax)
                action = v[action_index]

                action_prob_map["linear"][k_str] = {}
                action_prob_map["linear"][k_str]["action"] = action
                action_prob_map["linear"][k_str]["log_prob"] = log_prob.item()
                action_prob_map["linear"][k_str]["prob"] = prob.item()
                self.element_count += 1
                self.total_log_prob += log_prob
                # total_log_prob_list.append(log_prob)
        elif layer_type is 2:  # "multiheadattention"
            action_prob_map["multiheadattention"] = {}
            for k, v in self.architecture_map["multiheadattention"].items():
                # h_t, c_t = self.lstm1(state, (h_t, c_t))
                k_str = k  # + "@" + str(i)

                # hid = h_t.squeeze()
                linear_func = self.__getattr__(k_str)

                result = linear_func(hid)
                result_softmax = F.softmax(result, dim=-1)
                action_index, log_prob, prob = self.sample(result_softmax)
                action = v[action_index]

                action_prob_map["multiheadattention"][k_str] = {}
                action_prob_map["multiheadattention"][k_str]["action"] = action
                action_prob_map["multiheadattention"][k_str]["log_prob"] = log_prob.item()
                action_prob_map["multiheadattention"][k_str]["prob"] = prob.item()
                self.element_count += 1
                self.total_log_prob += log_prob
                # total_log_prob_list.append(log_prob)
        elif layer_type is 3:
            return action_prob_map
        elif layer_type is 4:
            action_prob_map["activate"] = {}
            for k, v in self.architecture_map["activate"].items():
                # h_t, c_t = self.lstm1(state, (h_t, c_t))
                k_str = k  # + "@" + str(i)

                # hid = h_t.squeeze()
                linear_func = self.__getattr__(k_str)

                result = linear_func(hid)
                result_softmax = F.softmax(result, dim=-1)
                action_index, log_prob, prob = self.sample(result_softmax)
                action = v[action_index]

                action_prob_map["activate"][k_str] = {}
                action_prob_map["activate"][k_str]["action"] = action
                action_prob_map["activate"][k_str]["log_prob"] = log_prob.item()
                action_prob_map["activate"][k_str]["prob"] = prob.item()
                self.element_count += 1
                self.total_log_prob += log_prob
                total_log_prob_list.append(log_prob)
        elif layer_type is 5:
            action_prob_map["dropout"] = {}
            for k, v in self.architecture_map["dropout"].items():
                # h_t, c_t = self.lstm1(state, (h_t, c_t))
                k_str = k  # + "@" + str(i)

                # hid = h_t.squeeze()
                linear_func = self.__getattr__(k_str)

                result = linear_func(hid)
                result_softmax = F.softmax(result, dim=-1)
                action_index, log_prob, prob = self.sample(result_softmax)
                action = v[action_index]

                action_prob_map["dropout"][k_str] = {}
                action_prob_map["dropout"][k_str]["action"] = action
                action_prob_map["dropout"][k_str]["log_prob"] = log_prob.item()
                action_prob_map["dropout"][k_str]["prob"] = prob.item()
                self.element_count += 1
                self.total_log_prob += log_prob
                total_log_prob_list.append(log_prob)
        elif layer_type is 6:
            action_prob_map["batch_norm"] = {}
            for k, v in self.architecture_map["batch_norm"].items():
                # h_t, c_t = self.lstm1(state, (h_t, c_t))
                k_str = k  # + "@" + str(i)

                # hid = h_t.squeeze()
                linear_func = self.__getattr__(k_str)

                result = linear_func(hid)
                result_softmax = F.softmax(result, dim=-1)
                action_index, log_prob, prob = self.sample(result_softmax)
                action = v[action_index]

                action_prob_map["batch_norm"][k_str] = {}
                action_prob_map["batch_norm"][k_str]["action"] = action
                action_prob_map["batch_norm"][k_str]["log_prob"] = log_prob.item()
                action_prob_map["batch_norm"][k_str]["prob"] = prob.item()
                self.element_count += 1
                self.total_log_prob += log_prob
                total_log_prob_list.append(log_prob)
        elif layer_type is 7:
            action_prob_map["pooling"] = {}
            for k, v in self.architecture_map["pooling"].items():
                # h_t, c_t = self.lstm1(state, (h_t, c_t))
                k_str = k  # + "@" + str(i)

                # hid = h_t.squeeze()
                linear_func = self.__getattr__(k_str)

                result = linear_func(hid)
                result_softmax = F.softmax(result, dim=-1)
                action_index, log_prob, prob = self.sample(result_softmax)
                action = v[action_index]

                action_prob_map["pooling"][k_str] = {}
                action_prob_map["pooling"][k_str]["action"] = action
                action_prob_map["pooling"][k_str]["log_prob"] = log_prob.item()
                action_prob_map["pooling"][k_str]["prob"] = prob.item()
                self.element_count += 1
                self.total_log_prob += log_prob
                # total_log_prob_list.append(log_prob)
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
            logging.error("unknow type", layer_type)
            return action_prob_map

        return action_prob_map


    def forward(self, state,conv_layer=2,linear_layer=2):
        # element_count = 0
        list_struct = []
        total_log_prob_list = []
        self.total_log_prob = 0
        self.element_count = 0

        list_type_prob = []
        log_prob = 0
        hid1 = self.transformer(state)
        # 用rnn这个先生成type 然后根据type生成对应种类的明细
        for i in range(self.max_layer):
            # h_t, c_t = self.lstm1(state, (h_t, c_t))
            hid = hid1.squeeze() #
            out1 = self.operator_common(hid)
            operator_count_str = "operator_count" + str(i)
            operator_count_linear = self.__getattr__(operator_count_str)
            output = operator_count_linear(out1)
            output = F.softmax(output, dim=-1)
            operator_dist = Categorical(output)
            operator_index = operator_dist.sample()
            operator_log_prob = operator_dist.log_prob(
                operator_index)  # log_prob returns the log of the probability density/mass function evaluated at the given sample value.
            self.element_count += 1
            self.total_log_prob += log_prob
            operator_prob = torch.exp(operator_log_prob)
            operator_index = operator_index.item()
            list_type_prob.append(("operator_type", operator_index, operator_prob.item()))
            # operator_index = 3
            # 0 add 1 subtract 2 concat 3 unary 4 multiply 5 skip
            if operator_index is 0:
                add_operator = {
                    "add":[]
                }
                for k in range(2):
                    single_action = self.unary_operator(state,i,total_log_prob_list,hid,list_type_prob)
                    if single_action == {}:
                        continue
                    add_operator["add"].append(single_action)
                if len(add_operator["add"]) == 1:
                    add_operator["unary"] = add_operator.pop("add")
                elif len(add_operator["add"]) == 0:
                    break
                list_struct.append(add_operator)
                # 每一层加个激活函数先不要加
                # action_prob_map = {}
                # operator = {"unary":[action_prob_map]}
                # h_t, c_t = self.lstm1(state, (h_t, c_t))
                # hid2 = c_t.squeeze()  #
                # result = self.type_activate(hid2)
                # result_softmax = F.softmax(result, dim=-1)
                # action_index, log_prob, prob = self.sample(result_softmax)
                # action = self.activation_functions_list[action_index]
                # action_prob_map["activate"] = {}
                # action_prob_map["activate"]["active"] = {}
                # action_prob_map["activate"]["active"]["action"] = action
                # action_prob_map["activate"]["active"]["log_prob"] = log_prob.item()
                # action_prob_map["activate"]["active"]["prob"] = prob.item()
                # self.element_count += 1
                # self.total_log_prob += log_prob
                # list_struct.append(operator)
            elif operator_index is 1:
                subtract_operator = {
                    "subtract":[],
                }
                for k in range(2):
                    single_action = self.unary_operator(state,i,total_log_prob_list,hid,list_type_prob)
                    if single_action == {}:
                        continue
                    subtract_operator["subtract"].append(single_action)
                if len(subtract_operator["subtract"]) == 1:
                    subtract_operator["unary"] = subtract_operator.pop("subtract")
                elif len(subtract_operator["subtract"]) == 0:
                    break

                list_struct.append(subtract_operator)
                # 每一层加个激活函数先不要加
                # action_prob_map = {}
                # operator = {"unary":[action_prob_map]}
                # h_t, c_t = self.lstm1(state, (h_t, c_t))
                # hid2 = c_t.squeeze()  #
                # result = self.type_activate(hid2)
                # result_softmax = F.softmax(result, dim=-1)
                # action_index, log_prob, prob = self.sample(result_softmax)
                # action = self.activation_functions_list[action_index]
                # action_prob_map["activate"]={}
                # action_prob_map["activate"]["active"] = {}
                # action_prob_map["activate"]["active"]["action"] = action
                # action_prob_map["activate"]["active"]["log_prob"] = log_prob.item()
                # action_prob_map["activate"]["active"]["prob"] = prob.item()
                # self.element_count += 1
                # self.total_log_prob += log_prob
                #
                # list_struct.append(operator)
            elif operator_index is 4:
                multiply_operator = {
                    "multiply": [],
                }
                for k in range(2):
                    single_action = self.unary_operator(state, i, total_log_prob_list, hid,
                                                                  list_type_prob)
                    if single_action == {}:
                        continue
                    multiply_operator["multiply"].append(single_action)
                if len(multiply_operator["multiply"]) == 1:
                    multiply_operator["unary"] = multiply_operator.pop("multiply")
                elif len(multiply_operator["multiply"]) == 0:
                    break

                list_struct.append(multiply_operator)
            elif operator_index is 5:
                skip_operator = {
                    "skip": {},

                }


                skip_layer_str = "skip_layer" + str(i)
                skip_layer_linear = self.__getattr__(skip_layer_str)
                output = skip_layer_linear(hid)
                output = F.softmax(output, dim=-1)
                action_dist = Categorical(output)
                action_index = action_dist.sample()
                type_log_prob = action_dist.log_prob(
                    action_index)  # log_prob returns the log of the probability density/mass function evaluated at the given sample value.
                type_prob = torch.exp(type_log_prob)
                skip_operator["skip"]["prob"] = type_prob.item()
                list_type_prob.append((skip_layer_str,action_index.item(), type_prob.item()))

                add_layer = action_index.item()
                self.element_count += 1
                self.total_log_prob += type_log_prob
                # to_layer = random.randint(i+1, self.max_layer)
                skip_operator["skip"]["from"] = i
                skip_operator["skip"]["to"] = i + add_layer+1
                list_struct.append(skip_operator)
            elif operator_index is 2:
                concat_operator = {
                    "concat": []
                }
                for k in range(2):
                    single_action = self.unary_operator(state, i, total_log_prob_list, hid,
                                                        list_type_prob)
                    if single_action == {}:
                        continue
                    concat_operator["concat"].append(single_action)
                if len(concat_operator["concat"]) == 1:
                    concat_operator["unary"] = concat_operator.pop("concat")
                elif len(concat_operator["concat"]) == 0:
                    break
                list_struct.append(concat_operator)

                # 每一层加个激活函数先不要加
                # action_prob_map = {}
                # operator = {"unary":[action_prob_map]}
                # h_t, c_t = self.lstm1(state, (h_t, c_t))
                # hid2 = c_t.squeeze()  #
                # result = self.type_activate(hid2)
                # result_softmax = F.softmax(result, dim=-1)
                # action_index, log_prob, prob = self.sample(result_softmax)
                # action = self.activation_functions_list[action_index]
                # action_prob_map["activate"] = {}
                # action_prob_map["activate"]["active"] = {}
                # action_prob_map["activate"]["active"]["action"] = action
                # action_prob_map["activate"]["active"]["log_prob"] = log_prob.item()
                # action_prob_map["activate"]["active"]["prob"] = prob.item()
                # self.element_count += 1
                # self.total_log_prob += log_prob
                #
                # list_struct.append(operator)
            elif operator_index is 3:
                operator = {

                }

                # unary operation

                action_prob_map = {}
                operator["unary"] = [action_prob_map]
                # h_t, c_t = self.lstm1(state, (h_t, c_t))
                # hid = c_t.squeeze() #
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
                list_type_prob.append((type_layer_str,action_index.item(),type_prob.item()))


                layer_type = action_index.item()
                self.element_count+=1
                self.total_log_prob += type_log_prob
                total_log_prob_list.append(type_log_prob)

                # 0 conv 1 linear 2 "multiheadattention" 3 end 4  activate function 6 batch norm 5 dropout 8 LayerNorm
                if layer_type is 0:
                    action_prob_map["conv"] = {}
                    for k, v in self.architecture_map["conv"].items():
                        # h_t, c_t = self.lstm1(state, (h_t, c_t))
                        k_str = k #+ "@" + str(i)
                        linear_func = self.__getattr__(k_str)
                        # hid = self.h_t.squeeze()
                        result = linear_func(hid)
                        result_softmax = F.softmax(result, dim=-1)
                        # action_prob = torch.sum(result_softmax, dim=0)
                        action_index,log_prob,prob = self.sample(result_softmax)
                        action = v[action_index]

                        action_prob_map["conv"][k_str] = {}
                        action_prob_map["conv"][k_str]["action"] = action
                        action_prob_map["conv"][k_str]["log_prob"] = log_prob.item()
                        action_prob_map["conv"][k_str]["prob"] = prob.item()
                        self.element_count += 1
                        self.total_log_prob += log_prob
                        total_log_prob_list.append(log_prob)
                elif layer_type is 1:
                    action_prob_map["linear"] = {}
                    for k, v in self.architecture_map["linear"].items():
                        # h_t, c_t = self.lstm1(state, (h_t, c_t))
                        k_str = k #+ "@" + str(i)
                        linear_func = self.__getattr__(k_str)
                        # hid = h_t.squeeze()
                        result = linear_func(hid)
                        result_softmax = F.softmax(result, dim=-1)
                        action_index, log_prob, prob = self.sample(result_softmax)
                        action = v[action_index]

                        action_prob_map["linear"][k_str] = {}
                        action_prob_map["linear"][k_str]["action"] = action
                        action_prob_map["linear"][k_str]["log_prob"] = log_prob.item()
                        action_prob_map["linear"][k_str]["prob"] = prob.item()
                        self.element_count += 1
                        self.total_log_prob += log_prob
                        total_log_prob_list.append(log_prob)
                elif layer_type is 2: #"multiheadattention"
                    action_prob_map["multiheadattention"] = {}
                    for k, v in self.architecture_map["multiheadattention"].items():
                        # h_t, c_t = self.lstm1(state, (h_t, c_t))
                        k_str = k #+ "@" + str(i)
                        # hid = h_t.squeeze()
                        linear_func = self.__getattr__(k_str)

                        result = linear_func(hid)
                        result_softmax = F.softmax(result, dim=-1)
                        action_index, log_prob, prob = self.sample(result_softmax)
                        action = v[action_index]

                        action_prob_map["multiheadattention"][k_str] = {}
                        action_prob_map["multiheadattention"][k_str]["action"] = action
                        action_prob_map["multiheadattention"][k_str]["log_prob"] = log_prob.item()
                        action_prob_map["multiheadattention"][k_str]["prob"] = prob.item()
                        self.element_count += 1
                        self.total_log_prob += log_prob
                        total_log_prob_list.append(log_prob)
                elif layer_type is 3:
                    break
                elif layer_type is 4:
                    action_prob_map["activate"] = {}
                    for k, v in self.architecture_map["activate"].items():
                        # h_t, c_t = self.lstm1(state, (h_t, c_t))
                        k_str = k #+ "@" + str(i)

                        # hid = h_t.squeeze()
                        linear_func = self.__getattr__(k_str)

                        result = linear_func(hid)
                        result_softmax = F.softmax(result, dim=-1)
                        action_index, log_prob, prob = self.sample(result_softmax)
                        action = v[action_index]

                        action_prob_map["activate"][k_str] = {}
                        action_prob_map["activate"][k_str]["action"] = action
                        action_prob_map["activate"][k_str]["log_prob"] = log_prob.item()
                        action_prob_map["activate"][k_str]["prob"] = prob.item()
                        self.element_count += 1
                        self.total_log_prob += log_prob
                        total_log_prob_list.append(log_prob)
                elif layer_type is 5:
                    action_prob_map["dropout"] = {}
                    for k, v in self.architecture_map["dropout"].items():
                        # h_t, c_t = self.lstm1(state, (h_t, c_t))
                        k_str = k  # + "@" + str(i)

                        # hid = h_t.squeeze()
                        linear_func = self.__getattr__(k_str)

                        result = linear_func(hid)
                        result_softmax = F.softmax(result, dim=-1)
                        action_index, log_prob, prob = self.sample(result_softmax)
                        action = v[action_index]

                        action_prob_map["dropout"][k_str] = {}
                        action_prob_map["dropout"][k_str]["action"] = action
                        action_prob_map["dropout"][k_str]["log_prob"] = log_prob.item()
                        action_prob_map["dropout"][k_str]["prob"] = prob.item()
                        self.element_count += 1
                        self.total_log_prob += log_prob
                        total_log_prob_list.append(log_prob)
                elif layer_type is 6:
                    action_prob_map["batch_norm"] = {}
                    for k, v in self.architecture_map["batch_norm"].items():
                        # h_t, c_t = self.lstm1(state, (h_t, c_t))
                        k_str = k  # + "@" + str(i)

                        # hid = h_t.squeeze()
                        linear_func = self.__getattr__(k_str)

                        result = linear_func(hid)
                        result_softmax = F.softmax(result, dim=-1)
                        action_index, log_prob, prob = self.sample(result_softmax)
                        action = v[action_index]

                        action_prob_map["batch_norm"][k_str] = {}
                        action_prob_map["batch_norm"][k_str]["action"] = action
                        action_prob_map["batch_norm"][k_str]["log_prob"] = log_prob.item()
                        action_prob_map["batch_norm"][k_str]["prob"] = prob.item()
                        self.element_count += 1
                        self.total_log_prob += log_prob
                        total_log_prob_list.append(log_prob)
                elif layer_type is 7:
                    action_prob_map["pooling"] = {}
                    for k, v in self.architecture_map["pooling"].items():
                        # h_t, c_t = self.lstm1(state, (h_t, c_t))
                        k_str = k  # + "@" + str(i)

                        linear_func = self.__getattr__(k_str)

                        result = linear_func(hid)
                        result_softmax = F.softmax(result, dim=-1)
                        action_index, log_prob, prob = self.sample(result_softmax)
                        action = v[action_index]

                        action_prob_map["pooling"][k_str] = {}
                        action_prob_map["pooling"][k_str]["action"] = action
                        action_prob_map["pooling"][k_str]["log_prob"] = log_prob.item()
                        action_prob_map["pooling"][k_str]["prob"] = prob.item()
                        self.element_count += 1
                        self.total_log_prob += log_prob
                        total_log_prob_list.append(log_prob)
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

                list_struct.append(operator)
        logging.info("type index  type prob" + str(list_type_prob))

        return list_struct,hid,self.total_log_prob,self.element_count,total_log_prob_list


    def sample(self,item_action_prob):
        action_dist = Categorical(item_action_prob)
        action_index = action_dist.sample()  # .unsqueeze(1)  这里就是根据概率进行采样
        log_prob = action_dist.log_prob(
            action_index)  # log_prob returns the log of the probability density/mass function evaluated at the given sample value.
        prob = torch.exp(log_prob)
        return action_index,log_prob,prob



    # def init_hidden(self):
    #     h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)
    #     c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=device)
    #
    #     return (h_t, c_t)








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
        self.conv1d_out_channels_list = [i for i in range(321) if i>9 and i%5==0] #[80,90,100,110]
        self.conv1d_kernel_size_list = [1,3,5,7,9]  #[7,5,1]
        self.drop_out_list = [0.02,0.05,0.1,0.15,0.2,0.07,0.12,0.16,0.23,0.25,0.3]
        self.conv_num_list = [1,2,3,4]
        self.linear_num_list = [1,2,3,4]
        self.conv_batch_norm_list = [0,1]
        self.batch_norm_list = [0, 1]

        self.linear40_40_out_features_list = [i for i in range(300) if i>2 and i%5==0]#[10,20,40,80]
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
                "nhead":[1,2,3],#因为dmodel% nhead要等于0
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
        self.optimizer = optim.SGD(self.network.parameters(),lr=0.00010)
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
        logging.info("reinfor loss "+str(loss)+" log_prob "+str(log_prob))
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
        action_prob_map,new_state,total_log_prob,element_count,total_log_prob_list = self.network(state,self.conv_num,self.linear_num) #torch.cuda.FloatTensor(
        # action_prob = torch.sum(item_action_prob,dim=0)   # To sum over all rows (i.e. for each column)  size = [1, ncol]
        if len(action_prob_map) is 0:
            return None,total_log_prob,None
        # regression_params = {}
        # self.total_log_prob = total_log_prob
        # self.sample_action(action_prob_map,regression_params)

        # 这里注意也要改
        avg_log_prob = total_log_prob/element_count


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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # optimizer 使用 Adam
        num_epoch = 60
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
                col = []
                # i = 0
                for i, item in enumerate(df["predict"]):
                    if item >= 0.5:
                        col.append(1)
                    else:
                        col.append(0)

                df["predict classification"] = col
                df["ground truth"] = target.cpu()
                col_truth = []
                # i = 0
                for i, item in enumerate(df["ground truth"]):
                    if item >= 0.9:
                        col_truth.append(1)
                    else:
                        col_truth.append(0)

                df["ground truth classification"] = col_truth
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







max_reward = -9999999
max_spearman = 0
max_pearson = 0
max_auc = 0

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



    # train_set_list = torch.utils.data.random_split(train_set,
    #                                                task_data_train)
    #
    # val_set_list = torch.utils.data.random_split(val_set, [300, 300, 300, 300, 300, 300, 300, 300, 300, 299])
    output_map={}
    reward_list = []
    output_map["reward_list"]=reward_list
    output_map["plot"]={}

    if path.exists("hf1_reward.json"):
        with open('hf1_reward.json', 'r') as openfile:
            # Reading from json file
            output_map = json.load(openfile)
            reward_list = output_map["reward_list"]


    agent = PolicyGradientAgent()

    agent.network.train()  # 訓練前，先確保 network 處在 training 模式

    NUM_BATCH = 5860000  # 總共更新 7600 次

    hidden_size = 64
    # state = torch.zeros(1, hidden_size, dtype=torch.float, device=device)
    state = torch.rand(1, 1, hidden_size).to(device)
    for batch in range(NUM_BATCH):
        # train_set, val_set = torch.utils.data.random_split(all_train_set, [12000, 2999])
        task = Task(HF1_train_set, HF1_test_set)
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
            tr_load = DataLoader(HF1_train_set, batch_size=512, shuffle=False)
            val_load = DataLoader(HF1_test_set, batch_size=512, shuffle=False)
            action_loss = task.train(model,tr_load)
            evaluate_loss,df = task.evaluate(model,val_load)
            rho, p = spearmanr(df["predict"], df["ground truth"])
            prho, pp = pearsonr(df["predict"], df["ground truth"])
            # fig, ax = plt.subplots()
            fpr, tpr, thresholds = metrics.roc_curve(df["ground truth classification"], df["predict"])
            # fpr1, tpr1, thresholds1 = metrics.roc_curve(df["ground truth classification"], df["predict classification"])
            roc_auc = metrics.auc(fpr, tpr)
            # auc = roc_auc_score(df["ground truth classification"],df["predict classification"])
            # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
            # estimator_name = 'example estimator')

            # display.plot(ax=ax)
            # plt.show()
            if math.isnan(rho):
                rho = 0
                p = 0

            logging.info("spearmanr " + str(rho) + " p " + str(p) + " pearsonr "+ str(prho) + " p " + str(pp)+ " auc "+str(roc_auc))
            #  以前main函数训练的结果记为baseline  reward 基于 baseline 来
            mean_loss = action_loss*0.15+evaluate_loss*0.85
            spearman_reward = rho * 500 + prho * 250 + roc_auc * 250
            struct_factor = 700/struct_len
            base_line = 700
            reward = spearman_reward - base_line - struct_factor
            reward_list.append(reward)
            if batch % 10 == 0:
                # Serializing json
                json_object = json.dumps(output_map, indent=4)

                # Writing to sample.json
                with open("hf1_reward.json", "w") as outfile:
                    outfile.write(json_object)


            global max_reward
            global max_spearman
            global max_pearson
            global max_auc

            if rho>max_spearman:
                max_spearman=rho
                logging.error("max_spearman:" + str(max_spearman) + " architecture:" + str(actionparam))

            if prho>max_pearson:
                max_pearson=prho
                logging.error("max_pearson:" + str(max_pearson) + " architecture:" + str(actionparam))

            if roc_auc > max_auc:
                max_auc = roc_auc
                output_map["plot"]["roc_auc"] = roc_auc
                output_map["plot"]["fpr"] = fpr.tolist()
                output_map["plot"]["tpr"] = tpr.tolist()
                plt.clf()
                plt.plot(fpr, tpr, color='darkorange',
                         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc="lower right")
                plt.savefig('image/hf1.png')
                json_object = json.dumps(output_map, indent=4)
                with open("hf1_reward.json", "w") as outfile:
                    outfile.write(json_object)
                logging.error("max_auc:" + str(max_auc) + " architecture:" + str(actionparam))

            if reward < -200 or math.isnan(reward):
                logging.error("bad architecture "+str(reward)+"fix to -200")
                reward = -200


            if reward > max_reward:
                max_reward = reward
                logging.error("max_reward:"+ str(max_reward) +" architecture:" + str(actionparam))


            if reward > 0:
                # reward = reward * 1.2
                logging.info("reward:***********************1.2")
                logging.info("reward:"+str(reward)+"mean_loss:"+str(mean_loss)+" action_loss"+str(action_loss)+"evaluate_loss"+str(evaluate_loss)+"spearmanr "+str(rho) + " p "+str(p))

            logging.info("reward:"+str(reward)+"struct_factor:"+str(struct_factor)+" spearman_reward"+str(spearman_reward)+"evaluate_loss"+str(evaluate_loss))
            agent.learn(reward,log_prob)
        except ValueError as vex:
            logging.info("!!!!!!!!!!"+str(vex)+"!!!!!!!!!!!!!!!!!!!!!!!!!!!!input <0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            reward = -200
            agent.learn(reward, log_prob)
        except Exception as ex:
            logging.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Except"+str(ex))
            reward = -200
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

