# -*- coding:utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Embedding, LSTM
from keras.utils import np_utils


def loda_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = loda_mnist();
    model = Sequential()
    # 2个卷积层
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))  # 第一层卷积
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 第一层池化
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(14, 14, 1)))  # 第二层卷积
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 第二层池化
    model.add(Dropout(0.25))  # 添加节点keep_prob
    # 2个全连接层
    model.add(Flatten())  # 将多维数据压成1维，方便全连接层操作
    model.add(Dense(1024, activation='relu'))  # 添加全连接层
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 训练模型
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)
    # 评估模型
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)