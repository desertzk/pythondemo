from PIL import Image
from IPython.display import display

# 我們看一下 Omniglot 的 dataset 長什麼樣子
for i in range(10, 20):
  im = Image.open("./data/hw13/Omniglot/images_background/Japanese_(hiragana).0/character13/0500_" + str (i) + ".png")
  display(im)

# Import modules we need
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import glob
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

'''
详细可以看视频 https://drive.google.com/file/d/1DjwXTpEVK__f5dmlkU4kUgaaTmFtHfIw/view
Step 2: 建立模型
以下我們就要開始建立核心的 MAML 模型 首先我們將需要的套件引入
'''

def ConvBlock(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2, stride=2))  # 原作者在 paper 裡是說她在 omniglot 用的是 strided convolution
    # 不過這裡我改成 max pool (mini imagenet 才是 max pool)
    # 這並不是你們在 report 第三題要找的 tip


def ConvBlockFunction(x, w, b, w_bn, b_bn):
    x = F.conv2d(x, w, b, padding=1)
    # 这里的norm的w_bn 全1 b_bn全 0
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


class Classifier(nn.Module):
    def __init__(self, in_ch, k_way):
        super(Classifier, self).__init__()
        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.logits = nn.Linear(64, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = nn.Flatten(x)
        x = self.logits(x)
        return x

    def functional_forward(self, x, params):
        '''
        Arguments:
        x: input images [batch, 1, 28, 28]
        params: 模型的參數，也就是 convolution 的 weight 跟 bias，以及 batchnormalization 的  weight 跟 bias
                這是一個 OrderedDict
        '''
        for block in [1, 2, 3, 4]:
            x = ConvBlockFunction(x, params[f'conv{block}.0.weight'], params[f'conv{block}.0.bias'],
                                  params.get(f'conv{block}.1.weight'), params.get(f'conv{block}.1.bias'))
        # 相当于np中的reshape可以拉平到linear model
        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits.weight'], params['logits.bias'])
        return x

# 這個函數是用來產生 label 的。在 n_way, k_shot 的 few-shot classification 問題中，每個 task 會有 n_way 個類別，每個類別k_shot張圖片。這是產生一個 n_way, k_shot 分類問題的 label 的函數
def create_label(n_way, k_shot):
    return torch.arange(n_way).repeat_interleave(k_shot).long()


# 我們試著產生 5 way 2 shot 的 label 看看
create_label(5, 2)
# tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

'''
接下來這裡是 MAML 的核心。演算法就跟原文完全一樣，這個函數做的事情就是用 "一個 meta-batch的 data" 更新參數。
這裡助教實作的是二階MAML(inner_train_step = 1)，對應老師投影片 meta learning p.13~p.18。如果要找一階的數學推導，在老師投影片 p.25。
'''

def MAML(model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_steps=1, inner_lr=0.4, train=True):
    """
    Args:
    x is the input omniglot images for a meta_step, shape = [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
    n_way: 每個分類的 task 要有幾個 class
    k_shot: 每個類別在 training 的時候會有多少張照片
    q_query: 在 testing 時，每個類別會用多少張照片 update
    """
    criterion = loss_fn
    task_loss = []  # 這裡面之後會放入每個 task 的 loss
    task_acc = []  # 這裡面之後會放入每個 task 的 loss
    # mate batch 32
    for meta_batch in x: # x [32,10,1,28,28] meta_batch [10,1,28,28]
        train_set = meta_batch[:n_way * k_shot]  # train_set 是我們拿來 update inner loop 參數的 data  [5,1,28,28]  取前5个
        val_set = meta_batch[n_way * k_shot:]  # val_set 是我們拿來 update outer loop 參數的 data   [5,1,28,28]  取后5个

        fast_weights = OrderedDict(
            model.named_parameters())  # 在 inner loop update 參數時，我們不能動到實際參數，因此用 fast_weights 來儲存新的參數 θ'

        for inner_step in range(inner_train_steps):  # 這個 for loop 是 Algorithm2 的 line 7~8
            # 實際上我們 inner loop 只有 update 一次 gradients，不過某些 task 可能會需要多次 update inner loop 的 θ'，
            # 所以我們還是用 for loop 來寫
            train_label = create_label(n_way, k_shot).cuda()
            logits = model.functional_forward(train_set, fast_weights)
            loss = criterion(logits, train_label)   #这里的loss 就是 一个task的loss
            grads = torch.autograd.grad(loss, fast_weights.values(),
                                        create_graph=True)  # 這裡是要計算出 loss 對 θ 的微分 (∇loss) 相当于ppt Meta1 30页 φ0 到 θm（助教备注algorithm中的θ'）这一步
            fast_weights = OrderedDict((name, param - inner_lr * grad)
                                       for ((name, param), grad) in
                                       zip(fast_weights.items(), grads))  #手写gradient decent 這裡是用剛剛算出的 ∇loss 來 update θ 變成 θ'

        val_label = create_label(n_way, q_query).cuda()
        # 这里的label 是用来干嘛的?
        logits = model.functional_forward(val_set, fast_weights)  # 這裡用 val_set 和 θ' 算 logit
        loss = criterion(logits, val_label)  # 這裡用 val_set 和 θ' 算 loss
        task_loss.append(loss)  # 把這個 task 的 loss 丟進 task_loss 裡面
        acc = np.asarray([torch.argmax(logits, -1).cpu().numpy() == val_label.cpu().numpy()]).mean()  # 算 accuracy
        task_acc.append(acc)

    model.train()
    optimizer.zero_grad()
    # 我們要用一整個 batch 的 loss 來 update θ (不是 θ') 把一个meta_batch task的loss平均一下然后做微分 这里是32个task loss的平均
    meta_batch_loss = torch.stack(task_loss).mean()
    if train:
        meta_batch_loss.backward()
        optimizer.step()              #这两句应该相当于第二次更新就是ppt Meta1   30页上从φ0 更新到 φ1
    task_acc = np.mean(task_acc)
    return meta_batch_loss, task_acc

# 定義 dataset。這個 dataset 會回傳某個 character 的 image，總共會有 k_shot+q_query 張，所以回傳的 tensor 大小是 [k_shot+q_query, 1, 28, 28]
class Omniglot(Dataset):
  def __init__(self, data_dir, k_way, q_query):
    self.file_list = [f for f in glob.glob(data_dir + "**/character*", recursive=True)]
    self.transform = transforms.Compose([transforms.ToTensor()])
    self.n = k_way + q_query # n_way: 每個分類的 task 要有幾個 class  q_query: 在 testing 時，每個類別會用多少張照片 update
  def __getitem__(self, idx):
    sample = np.arange(20)
    np.random.shuffle(sample) # 這裡是為了等一下要 random sample 出我們要的 character
    img_path = self.file_list[idx]
    img_list = [f for f in glob.glob(img_path + "**/*.png", recursive=True)]
    img_list.sort()
    print(img_list)
    imgs = [self.transform(Image.open(img_file)) for img_file in img_list]
    print(sample[:self.n])
    imgs = torch.stack(imgs)[sample[:self.n]] # 每個 character，取出 k_way + q_query（本例中是2） 個 这句话其实是取出其中的self.n张图  imgs[0].numpy()可以看到图象
    return imgs
  def __len__(self):
    return len(self.file_list)

# 定義 hyperparameter
n_way = 5  # 每個分類的 task 要有幾個 5
k_shot = 1   # 代表 1个train set
# 设定了 n_way=5 k_shot=1的意思就是说 有5种类别的字符每个字符只有1个例子
q_query = 1
# q_query = 1 代表 取一张图代表 test set
inner_train_steps = 1
inner_lr = 0.4
meta_lr = 0.001
meta_batch_size = 32    # 代表有32个 task
max_epoch = 40
eval_batches = test_batches = 20
train_data_path = './data/hw13/Omniglot/images_background/'
test_data_path = './data/hw13/Omniglot/images_evaluation/'

# 初始化 dataloader
dataset = Omniglot(train_data_path, k_shot, q_query)
train_set, val_set = torch.utils.data.random_split(Omniglot(train_data_path, k_shot, q_query), [3200,656])
train_loader = DataLoader(train_set,
                          batch_size = n_way, # 這裡的 batch size 並不是 meta batch size, 而是一個 task裡面會有多少不同的
                                              # characters，也就是 few-shot classifiecation 的 n_way
                          num_workers = 0,
                          shuffle = True,
                          drop_last = True)
val_loader = DataLoader(val_set,
                          batch_size = n_way,
                          num_workers = 0,
                          shuffle = True,
                          drop_last = True)
test_loader = DataLoader(Omniglot(test_data_path, k_shot, q_query),
                          batch_size = n_way,
                          num_workers = 0,
                          shuffle = True,
                          drop_last = True)
train_iter = iter(train_loader)
val_iter = iter(val_loader)
test_iter = iter(test_loader)

# 初始化 model 和 optimizer
meta_model = Classifier(1, n_way).cuda()
# meta_lr 这个就是out learning rate 相当于
optimizer = torch.optim.Adam(meta_model.parameters(), lr = meta_lr)
loss_fn = nn.CrossEntropyLoss().cuda()

# 這是一個用來抓一個 meta-batch 的 data 出來的 function
def get_meta_batch(meta_batch_size, k_shot, q_query, data_loader, iterator):
  data = []
  for _ in range(meta_batch_size):
    try:
      task_data = iterator.next()  # 一筆 task_data 就是一個 task 裡面的 data，大小是 [n_way本例中是 5, k_shot+q_query, 1, 28, 28]
    except StopIteration:
      iterator = iter(data_loader)
      task_data = iterator.next()
    train_data = task_data[:, :k_shot].reshape(-1, 1, 28, 28) # 取出 5组
    val_data = task_data[:, k_shot:].reshape(-1, 1, 28, 28)
    task_data = torch.cat((train_data, val_data), 0)
    data.append(task_data)
  return torch.stack(data).cuda(), iterator

# 開始 train!!!
for epoch in range(max_epoch):
  print("Epoch %d" %(epoch))
  train_meta_loss = []
  train_acc = []
  for step in tqdm(range(len(train_loader) // (meta_batch_size))): # 這裡的 step 是一次 meta-gradinet update step
    x, train_iter = get_meta_batch(meta_batch_size, k_shot, q_query, train_loader, train_iter)
    meta_loss, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn)
    train_meta_loss.append(meta_loss.item())
    train_acc.append(acc)
  print("  Loss    : ", np.mean(train_meta_loss))
  print("  Accuracy: ", np.mean(train_acc))

  # 每個 epoch 結束後，看看 validation accuracy 如何
  # 助教並沒有做 early stopping，同學如果覺得有需要是可以做的
  val_acc = []
  for eval_step in tqdm(range(len(val_loader) // (eval_batches))):
    x, val_iter = get_meta_batch(eval_batches, k_shot, q_query, val_loader, val_iter)
    _, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step = 3, train = False) # testing時，我們更新三次 inner-step
    val_acc.append(acc)
  print("  Validation accuracy: ", np.mean(val_acc))


# 測試訓練結果。這就是report上要回報的 test accuracy。
test_acc = []
for test_step in tqdm(range(len(test_loader) // (test_batches))):
  x, val_iter = get_meta_batch(test_batches, k_shot, q_query, test_loader, test_iter)
  _, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step = 3, train = False) # testing時，我們更新三次 inner-step
  test_acc.append(acc)
print("  Testing accuracy: ", np.mean(test_acc))
