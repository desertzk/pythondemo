import tensorflow as tf
import numpy as np

# The inputs are 128-length vectors with 10 timesteps, and the batch size
# is 4.
input_shape = (1, 3, 3)
x = tf.random.normal(input_shape)
print(x)

print('x.shape: ', x.shape)
print('input shape parameter value to Conv1D: ', input_shape[1:])
y = tf.keras.layers.Conv1D(32, 3, activation='relu',input_shape=input_shape[1:])(x)
print('y.shape: ',y.shape)
print('y:\n',y)
# prepare a filter such that
# W is all ones
# b is zero
myWeights=(np.ones((kernelSize,student,1)), np.zeros(1,))
model.get_layer(index=1).set_weights(myWeights)