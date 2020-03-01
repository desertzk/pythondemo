import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 创建一张图包含了一组op和tensor,上下文环境
# op:只要使用tensorflow的API定义的函数都是OP
# tensor：就指代的是数据

# g = tf.Graph()
#
# print(g)
# with g.as_default():
#     c = tf.constant(11.0)
#     print(c.graph)
#
# # 实现一个加法运算
# a = tf.constant(5.0)
# b = tf.constant(6.0)
#
# sum1 = tf.add(a, b)
#
# # 默认的这张图，相当于是给程序分配一段内存
# graph = tf.get_default_graph()
#
# print(graph)
#
# # 不是op不能运行
# var1 = 2.0
# # var2 = 3
# # sum2 = var1 + var2
#
# # 有重载的机制,默认会给运算符重载成op类型
# sum2 = a + var1
#
# print(sum2)
#
# # s = tf.Session()
# #
# # s.run()
# # s.run()
# # s.close()
#
# # 只能运行一个图， 可以在会话当中指定图去运行
# # 只要有会话的上下文环境，就可以使用方便eval()
#
# # 训练模型
# # 实时的提供数据去进行训练
#
# # placeholder是一个占位符,feed_dict一个字典
# plt = tf.placeholder(tf.float32, [2, 3, 4])
#
# print(plt)
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     # print(sess.run(plt, feed_dict={plt: [[1, 2, 3], [4, 5, 36], [2, 3, 4]]}))
#     # print(sum1.eval())
#     print(a.graph)
#     print("---------")
#     print(a.shape)
#     print(plt.shape)
#     print("-------")
#     print(a.name)
#     print("-------")
#     print(a.op)

# tensorflow:打印出来的形状表示
# 0维：()   1维:(5)  2维：(5,6)   3维：(2,3,4)

# 形状的概念
# 静态形状和动态性状
# 对于静态形状来说，一旦张量形状固定了，不能再次设置静态形状, 不能夸维度修改 1D->1D 2D->2D
# 动态形状可以去创建一个新的张量,改变时候一定要注意元素数量要匹配  1D->2D  1->3D
#
# plt = tf.placeholder(tf.float32, [None, 2])
#
# print(plt)
#
# plt.set_shape([3, 2, 1])
#
# print(plt)
#
# # plt.set_shape([2, 3]) # 不能再次修改
#
# plt_reshape = tf.reshape(plt, [3, 3])
#
# print(plt_reshape)
#
# with tf.Session() as sess:
#     pass


# 变量op
# 1、变量op能够持久化保存，普通张量op是不行的
# 2、当定义一个变量op的时候，一定要在会话当中去运行初始化
# 3、name参数：在tensorboard使用的时候显示名字，可以让相同op名字的进行区分

# a = tf.constant(3.0, name="a")
#
# b = tf.constant(4.0, name="b")
#
# c = tf.add(a, b, name="add")
#
# var = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0), name="variable")
#
# print(a, var)
#
# # 必须做一步显示的初始化op
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     # 必须运行初始化op
#     sess.run(init_op)
#
#     # 把程序的图结构写入事件文件, graph:把指定的图写进事件文件当中
#     filewriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)
#
#     print(sess.run([c, var]))

# 1、训练参数问题:trainable
# 学习率和步数的设置：

# 2、添加权重参数，损失值等在tensorboard观察的情况 1、收集变量2、合并变量写入事件文件

# 定义命令行参数
# 1、首先定义有哪些参数需要在运行时候指定
# 2、程序当中获取定义命令行参数

# 第一个参数：名字，默认值，说明
# tf.app.flags.DEFINE_integer("max_step", 100, "模型训练的步数")
# tf.app.flags.DEFINE_string("model_dir", " ", "模型文件的加载的路径")
#
# # 定义获取命令行参数名字
# FLAGS = tf.app.flags.FLAGS


def myregression():
    """
    自实现一个线性回归预测
    :return: None
    """
    with tf.variable_scope("data"):
        # 1、准备数据，x 特征值 [100, 1]   y 目标值[100]
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")

        # 矩阵相乘必须是二维的
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("model"):
        # 2、建立线性回归模型 1个特征，1个权重， 一个偏置 y = x w + b
        # 随机给一个权重和偏置的值，让他去计算损失，然后再当前状态下优化
        # 用变量定义才能优化
        # trainable参数：指定这个变量能跟着梯度下降一起优化
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name="w")
        bias = tf.Variable(0.0, name="b")

        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope("loss"):
        # 3、建立损失函数，均方误差
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    with tf.variable_scope("optimizer"):
        # 4、梯度下降优化损失 leaning_rate: 0 ~ 1, 2, 3,5, 7, 10
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 1、收集tensor
    tf.summary.scalar("losses", loss)
    tf.summary.histogram("weights", weight)

    # 定义合并tensor的op
    merged = tf.summary.merge_all()

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 定义一个保存模型的实例
    saver = tf.train.Saver()

    # 通过会话运行程序
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 打印随机最先初始化的权重和偏置
        print("随机初始化的参数权重为：%f, 偏置为：%f" % (weight.eval(), bias.eval()))

        # 建立事件文件
        filewriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)

        # 加载模型，覆盖模型当中随机定义的参数，从上次训练的参数结果开始
        # if os.path.exists("./tmp/ckpt/checkpoint"):
        #     saver.restore(sess, FLAGS.model_dir)

        # 循环训练 运行优化
        for i in range(100):

            sess.run(train_op)

            # 运行合并的tensor
            summary = sess.run(merged)

            filewriter.add_summary(summary, i)

            print("第%d次优化的参数权重为：%f, 偏置为：%f" % (i, weight.eval(), bias.eval()))

        # saver.save(sess, FLAGS.model_dir)
    return None


if __name__ == "__main__":
    myregression()













