import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)  # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))  # 这是答案~

# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]  # first 160 data points
X_test, Y_test = X[160:], Y[160:]  # last 40 data points
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')
# training
print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

import numpy as np

np.random.seed(1337)
# 作者想把每次的随机数设置成一样的，这样每个人的实验结果都相同，很有成就感。
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt  # 可视化的模块

# 1 数据准备，造点数据
X = np.linspace(-1, 1, 200)  # 200个有序数字
np.random.shuffle(X)  # 把X顺序打乱
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))  # 这是“答案” 看看keras能不能找到“0.5和2"

## 画出数据
plt.scatter(X, Y)
plt.show()

## 把数据分为，训练和测试两部分
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# 2 用keras构建神经网络，这次就一层，而且就一个神经元，有点搓啊~
model = Sequential()  # 建立模型
model.add(Dense(input_dim=1, output_dim=1))  # 添加一层，一个输入一个输出

# 3 激活模型
model.compile(loss='mse', optimizer='sgd')
## 这样2和3 一共三行代码就实现了，所以比tensorflow简单多了

# 4 训练模型
print("----------------------training--------------------------")
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print("train cost:", cost)

# 5 检验模型
print("--------------testing-------------------")
cost = model.evaluate(X_test, Y_test, batch_size=40)
print("test cost:", cost)
W, b = model.layers[0].get_weights()
print("Weights:", W, "\nbaise:", b)

# 6 结果可视化
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()