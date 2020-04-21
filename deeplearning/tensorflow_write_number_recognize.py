import tensorflow as tf
import numpy as np

print(tf.__version__)
#4.Import the minist dataset by numpy offline
def load_mnist():
    #define the directory where mnist.npz is(Please watch the '\'!)
    path = r'F:\learning\machineLearning\forward_progression\mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'],f['y_train']
    x_test, y_test = f['x_test'],f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

print(train_images)