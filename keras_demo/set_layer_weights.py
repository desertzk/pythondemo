import numpy as np
np.random.seed(1234)
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.models import Model
print("Building Model...")
input = np.asarray([
    [[
    [1.,2.,3.],
    [4.,5.,6.],
    [7.,8.,9.]
    ]]
    ])
print(input.shape)
# 維度互換
input_mat = np.transpose(input,(0,3,2,1))
print(input_mat.shape)
inp = Input(shape=(3,3,1))
# inp = Input(shape=(1,3,3))
output = Convolution2D(1, 3, use_bias=False)(inp)
model_network = Model(inputs=inp, outputs=output)
print("Weights before change:")
print (model_network.layers[1].get_weights())
print (model_network.layers[1].get_weights()[0].shape)
w_e = np.asarray([
    [[[
    [0,0,0],
    [0,2,0],
    [0,0,0]
    ]]]
    ])
w = np.transpose(w_e,(0,3,4,1,2))
print(w.shape)
# w = np.asarray(
#     [[
#         [[0]],
#         [[0]],
#         [[0]],
#     ],[
#         [[0]],
#         [[2]],
#         [[0]],
#     ],[
#         [[0]],
#         [[0]],
#         [[0]],
#     ]]
#     )
print(w.shape)
model_network.layers[1].set_weights(w)
print("Weights after change:")
print(model_network.layers[1].get_weights())
print("Input:")
print(input_mat)
print("Output:")
print(model_network.predict(input_mat))