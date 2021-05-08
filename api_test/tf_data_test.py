import tensorflow as tf

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)

dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

for elem in dataset:
  print(elem.numpy())

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
print(dataset1.element_spec)

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

print(dataset2.element_spec)

train, test = tf.keras.datasets.fashion_mnist.load_data()
images, labels = train
images = images/255

# print("image example")
# print(images[0])

print("label example")
print(labels[0])

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
print(dataset)

fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")

dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())
print(parsed)

dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=(None,))
for batch in dataset.take(2):
  print(batch.numpy())
  print()
