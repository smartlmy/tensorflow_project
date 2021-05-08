import tensorflow as tf

tf.compat.v1.disable_eager_execution()

x = tf.Variable(initial_value=tf.compat.v1.random_normal(shape=[2,3], mean=0, stddev=1), dtype='float32')
w = tf.Variable(initial_value=tf.compat.v1.random_normal(shape=[2,3], mean=0, stddev=1), dtype='float32')
y = w*x

opt = tf.compat.v1.train.GradientDescentOptimizer(0.1)
grad = opt.compute_gradients(y, [w,x])
z = tf.concat(list(grad[0]), axis=1)
max_element_z = tf.math.reduce_max(z)
avg_z = tf.math.reduce_mean(z)
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for grad, var in grad:
        print(sess.run(var.name))
        print(sess.run(grad))
        print(sess.run(var))