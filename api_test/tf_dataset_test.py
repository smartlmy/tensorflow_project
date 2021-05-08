import tensorflow as tf

filenames = ["/var/data/file1.txt", "/var/data/file2.txt",
             "/var/data/file3.txt", "/var/data/file4.txt"]
dataset = tf.data.Dataset.from_tensor_slices(filenames)
def parse_fn(filename):
  return tf.data.Dataset.range(10)

dataset = dataset.interleave(lambda x:
    tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
    cycle_length=4, block_length=16)

#
dataset = tf.data.Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
# NOTE: New lines indicate "block" boundaries.
dataset = dataset.interleave(
    lambda x: tf.data.Dataset.from_tensors(x).repeat(6),
    cycle_length=2, block_length=4)
print(list(dataset.as_numpy_iterator()))

