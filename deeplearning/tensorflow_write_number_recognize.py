import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.model_selection import ShuffleSplit
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
print(train_images.shape)
train_labels = np.eye(10)[train_labels]
train_images_2dim = np.resize(train_images, (train_images.shape[0],748))
train_images_2dim = np.divide(train_images_2dim,255.0)
# 建立输入数据占位符
x = tf.placeholder(tf.float32, [None, 748])
y_true = tf.placeholder(tf.float32, [None, 10])
#初始化权重和偏置
W = tf.Variable(initial_value = tf.random_normal(shape=(748, 10)))
b = tf.Variable(initial_value = tf.random_normal(shape=([10])))

# 输出结果
y_predict = tf.matmul(x, W) + b

loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss_func)
# 计算准确率
correct_prediction = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# part_of_train = train_images_2dim

init = tf.global_variables_initializer()

session = tf.Session()
train_data = train_images_2dim
ss = ShuffleSplit(n_splits=2000, train_size=50)
ss.get_n_splits(train_data, train_labels)
history = [(0, np.nan, 10)] # Initial Error Measures
for step, (idx, _) in enumerate(ss.split(train_data,train_labels), start=1):
    fd = {x:train_data[idx], y_true:train_labels[idx]}
    session.run(optimizer, feed_dict=fd)
    if step%100 == 0:
        # fd = {tf_data:valid_data, tf_labels:valid_labels}
        # valid_loss, valid_accuracy = session.run([tf_loss, tf_acc], feed_dict=fd)
        op, loss, accuracy_value =session.run([optimizer,loss_func,accuracy],feed_dict = fd)
        history.append((step, loss, accuracy_value))
        print('Step %i \t Valid. Acc. = %f \n'%(step, accuracy_value))



# with tf.Session() as sess:
#     sess.run(init)
#
#     # train = train_images_2dim.next_batch(100)
#     # labels = onehot_labels.next_batch(100)
#     print("before loss %f"%(sess.run(loss_func,feed_dict = {x:train_images_2dim,y_true:onehot_labels})))
#
#     for i in range(3000):
#         op,loss,accuracy_value=sess.run([optimizer,loss_func,accuracy],feed_dict = {x:train_images_2dim,y_true:onehot_labels})
#         print("%d th training  loss is %f    accuracy_value %f"%(i+1,loss,accuracy_value))
#         # print("W = "+ str(W)+ " b = "+str(b))



# print(train_images)