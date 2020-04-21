
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))

# from tensorflow.examples.tutorials.mnist import input_data
#
# # 输入数据
# mnist = input_data.read_data_sets('FLAGS.data_dir', one_hot=True)
