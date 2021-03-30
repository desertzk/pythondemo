# utils.py
# 這個block用來先定義一些等等常用到的函式
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec



def load_training_data(path='data/hw4/training_label.txt'):
    # 把training時需要的data讀進來
    # 如果是'training_label.txt'，需要讀取label，如果是'training_nolabel.txt'，不需要讀取label
    if 'training_label' in path:
        with open(path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r',encoding="utf-8") as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x

def load_testing_data(path='data/hw4/testing_data'):
    # 把testing時需要的data讀進來
    with open(path, 'r',encoding="utf-8") as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X

def evaluation(outputs, labels):
    #outputs => probability (float)
    #labels => labels
    outputs[outputs>=0.5] = 1 # 大於等於0.5為有惡意
    outputs[outputs<0.5] = 0 # 小於0.5為無惡意
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

from torch import nn
from gensim.models import Word2Vec

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    def get_w2v_model(self):
        # 把之前訓練好的word to vec 模型讀進來
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size  #每一个词都变成250维的向量，这个向量里包含了每个词之间的关系
    def add_embedding(self, word):
        # 把word加進embedding，並賦予他一個隨機生成的representation vector
        # word只會是"<PAD>"或"<UNK>"
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 取得訓練好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 製作一個 word2idx 的 dictionary
        # 製作一個 idx2word 的 list
        # 製作一個 word2vector 的 list
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['魯'] = 1
            #e.g. self.index2word[1] = '魯'
            #e.g. self.vectors[1] = '魯' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])   #每一个词都变成250维的向量，这个向量里包含了每个词之间的关系
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 將"<PAD>"跟"<UNK>"加進embedding裡面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    def pad_sequence(self, sentence):
        # 將每個句子變成一樣的長度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    def sentence_word2idx(self):
        # 把句子裡面的字轉成相對應的index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 將每個句子變成一樣的長度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    def labels_to_tensor(self, y):
        # 把labels轉成tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)



def train_word2vec(x):
    # 訓練word to vector 的 word embedding
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model


import torch
from torch.utils import data


class TwitterDataset(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)

    __len__ will return the number of data
    """

    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


# 這個block是要拿來訓練的模型
import torch
from torch import nn
class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix住，如果fix_embedding為False，在訓練過程中，embedding也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, 1),
                                         nn.Sigmoid() )
    def forward(self, inputs):
        inputs = self.embedding(inputs) #通过index 转化为 embedding 数据
        x, _ = self.lstm(inputs, None) #batch , word size, word embedding vector
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        x = x[:, -1, :]
        x = self.classifier(x)
        return x






def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train() # 將model的模式設為train，這樣optimizer就可以更新model的參數
    criterion = nn.BCELoss() # 定義損失函數，這裡我們使用binary cross entropy loss
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr) # 將模型的參數給optimizer，並給予適當的learning rate
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # 這段做training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long) # device為"cuda"，將inputs轉成torch.cuda.LongTensor
            labels = labels.to(device, dtype=torch.float) # device為"cuda"，將labels轉成torch.cuda.FloatTensor，因為等等要餵進criterion，所以型態要是float
            optimizer.zero_grad() # 由於loss.backward()的gradient會累加，所以每次餵完一個batch後需要歸零
            outputs = model(inputs) # 將input餵給模型 这里是embedding index以后的数据
            outputs = outputs.squeeze() # 去掉最外面的dimension，好讓outputs可以餵進criterion()
            loss = criterion(outputs, labels) # 計算此時模型的training loss
            loss.backward() # 算loss的gradient
            optimizer.step() # 更新訓練模型的參數
            correct = evaluation(outputs, labels) # 計算此時模型的training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))

        # 這段做validation
        model.eval() # 將model的模式設為eval，這樣model的參數就會固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long) # device為"cuda"，將inputs轉成torch.cuda.LongTensor
                labels = labels.to(device, dtype=torch.float) # device為"cuda"，將labels轉成torch.cuda.FloatTensor，因為等等要餵進criterion，所以型態要是float
                outputs = model(inputs) # 將input餵給模型
                outputs = outputs.squeeze() # 去掉最外面的dimension，好讓outputs可以餵進criterion()
                loss = criterion(outputs, labels) # 計算此時模型的validation loss
                correct = evaluation(outputs, labels) # 計算此時模型的validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                # 如果validation的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                best_acc = total_acc
                #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')
        model.train() # 將model的模式設為train，這樣optimizer就可以更新model的參數（因為剛剛轉成eval模式）


import torch
from torch import nn
import torch.optim as optim



def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1  # 大於等於0.5為負面
            outputs[outputs < 0.5] = 0  # 小於0.5為正面
            ret_output += outputs.int().tolist()

    return ret_output



if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = load_training_data('data/hw4/training_label.txt')
    train_x_no_label = load_training_data('data/hw4/training_nolabel.txt')

    print("loading testing data ...")
    test_x = load_testing_data('data/hw4/testing_data.txt')

    model = train_word2vec(train_x + train_x_no_label + test_x)

    print("saving model ...")
    # model.save(os.path.join(path_prefix, 'model/w2v_all.model'))
    model.save(os.path.join(".", 'data/hw4/w2v_all.model'))


    # 通過torch.cuda.is_available()的回傳值進行判斷是否有使用GPU的環境，如果有的話device就設為"cuda"，沒有的話就設為"cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 處理好各個data的路徑
    train_with_label = os.path.join(".", 'data/hw4/training_label.txt')
    train_no_label = os.path.join(".", 'data/hw4/training_nolabel.txt')
    testing_data = os.path.join(".", 'data/hw4/testing_data.txt')

    w2v_path = os.path.join(".", 'data/hw4/w2v_all.model')  # 處理word to vec model的路徑

    # 定義句子長度、要不要固定embedding、batch大小、要訓練幾個epoch、learning rate的值、model的資料夾路徑
    sen_len = 30
    fix_embedding = True  # fix embedding during training
    batch_size = 128
    epoch = 5
    lr = 0.001
    # model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
    model_dir = "data/hw4/"  # model directory for checkpoint model

    print("loading data ...")  # 把'training_label.txt'跟'training_nolabel.txt'讀進來
    train_x, y = load_training_data(train_with_label)
    train_x_no_label = load_training_data(train_no_label)

    # 對input跟labels做預處理
    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)

    # 製作一個model的對象
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=250, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
    model = model.to(device)  # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)

    # 把data分為training data跟validation data(將一部份training data拿去當作validation data)
    X_train, X_val, y_train, y_val = train_x[:190000], train_x[190000:], y[:190000], y[190000:]

    # 把data做成dataset供dataloader取用
    train_dataset = TwitterDataset(X=X_train, y=y_train)
    val_dataset = TwitterDataset(X=X_val, y=y_val)

    # 把data 轉成 batch of tensors
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8)

    # 開始訓練
    training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)

    # 開始測試模型並做預測
    print("loading testing data ...")
    test_x = load_testing_data(testing_data)
    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X=test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 8)
    print('\nload model ...')
    model = torch.load(os.path.join(model_dir, 'ckpt.model'))
    outputs = testing(batch_size, test_loader, model, device)

    # 寫到csv檔案供上傳kaggle
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
    print("save csv ...")
    tmp.to_csv(os.path.join(".", 'predict.csv'), index=False)
    print("Finish Predicting")