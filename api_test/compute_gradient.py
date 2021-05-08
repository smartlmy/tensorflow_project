import tensorflow as tf

w = tf.Variable([2., 5.])
u = tf.Variable([7., 9.])

x = 2 * w + 3 * u

with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run()
    optimizer = tf.compat.v1.train.AdamOptimizer(0.1)
    gvs = optimizer.compute_gradients(x)
    print(sess.run(gvs))