import tensorflow as tf
import numpy as np

def apply_discount(x):
    a = tf.constant(2.0)
    return tf.math.log(a) / tf.math.log(a + tf.cast(x, tf.float32))

logit_vals = tf.random.normal([2, 5], 0, 1, tf.float32)
print(logit_vals)
a = tf.argsort(logit_vals, axis=1)
print("arg sort first")
print(a)
print("arg sort second")
b = tf.argsort(a, axis=1)
print(b)
ranks = 4-b
print(ranks)
print(np.expand_dims(ranks[:, 0], axis=1))
print(tf.cast(ranks[:, 1:], tf.float32) + 2.0)

discounted_non_booking = apply_discount(ranks[:, 1:])
discounted_booking = apply_discount(np.expand_dims(ranks[:, 0], axis=1))
print(discounted_non_booking)
print(discounted_booking)
discounted_weights = np.abs(discounted_booking - discounted_non_booking)
print(discounted_weights)