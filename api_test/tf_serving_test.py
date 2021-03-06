import os
import shutil

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def add_layer(inputs, input_size, output_size, activation_function=None):
    weights = tf.Variable(tf.random.normal([input_size, output_size]))
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases  # WX + b
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs

# 造一些随机输入数据
num_points = 30000  # 总数据条数
feature_number = 100  # 每条输入数据有100个feature
# num_points个输入数据,每个有feature_number个feature,即输入数据的维度是(num_points,feature_number)
x_data = np.random.rand(num_points, feature_number)
y_data = np.random.randint(0, 2, (num_points, 1))  # nx1的数组, 每一行为1个数(0或1)

# 用于接收输入的Tensor
x_actual = tf.compat.v1.placeholder(tf.float32, [None, feature_number], name="myInput")
y_actual = tf.compat.v1.placeholder(tf.float32, [None, 1], name="myOutput")

# 隐层1
l1 = add_layer(x_actual, feature_number, 32, activation_function=tf.nn.relu)
# 隐层2
l2 = add_layer(l1, 32, 64, activation_function=tf.nn.tanh)
# 隐层3
l3 = add_layer(l2, 64, 32, activation_function=tf.nn.relu)

# 输出层
y_predict = add_layer(l3, 32, 1, activation_function=tf.nn.sigmoid)
# 损失函数
loss = -tf.reduce_mean(y_actual * tf.math.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))
# 优化器
train_step = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)

init = tf.compat.v1.global_variables_initializer()
# 迭代次数
a = 1
num_iterations = 1000
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for i in range(num_iterations):
        # 训练模型
        sess.run(train_step, feed_dict={x_actual: x_data, y_actual: y_data})
        if i % 500 == 0:
            prediction_value = sess.run(y_predict, feed_dict={x_actual: x_data})
            print(sess.run(loss, feed_dict={x_actual: x_data, y_actual: y_data}))
    # 训练完成后,以SavedModel格式保存模型文件
    model_output_dir = "../model/tf_serving_test/1"
    if os.path.exists(model_output_dir):  # 目录存在
        shutil.rmtree(model_output_dir)  # 删除原目录
    tf.compat.v1.saved_model.simple_save(
        sess, model_output_dir, inputs={"myInput": x_actual}, outputs={"myOutput": y_predict})

    # 做5次预测(测试一下)
    for i in range(5):
        x_input = np.random.rand(1, feature_number)  # 1表示输入一条数据
        feed_dict = {x_actual: x_input}
        result = sess.run(y_predict, feed_dict)
        print('prediction result: %f' % result)