import tensorflow as tf

# dense output summary
def add_dense_output_summary(collection_name, summary_prefix="DenseOutput/", contain_string=""):
 variables = tf.get_collection(collection_name)
 for x in variables:
   if contain_string in x.name:
     try:
       logging.info("add dense_output %s to summary with shape %s" % (str(x.name), str(x.shape)))
       if len(x.shape) == 3:
         tf.summary.scalar(name=summary_prefix + "Norm2/" + x.name.replace(":", "_"),
                           tensor=tf.reduce_mean(tf.norm(x)))
         tf.summary.histogram(name=summary_prefix + "AbsHistMax/" + x.name.replace(":", "_"),
                              values=tf.reduce_max(tf.abs(x), axis=-1))
         mean, variance = tf.nn.moments(x, axes=[0, 1])
         tf.summary.scalar(name=summary_prefix + "Mean/" + x.name.replace(":", "_"),
                           tensor=tf.reduce_mean(mean))
         tf.summary.scalar(name=summary_prefix + "Variance/" + x.name.replace(":", "_"),
                           tensor=tf.reduce_mean(variance))
         tf.summary.histogram(name=summary_prefix + "PosR/" + x.name.replace(":", "_"),
                              values=greater_zero_histogram(x))
       elif x.shape[1] > 1:
         tf.summary.scalar(name=summary_prefix + "Norm2/" + x.name.replace(":", "_"),
                           tensor=tf.reduce_mean(tf.norm(x, axis=-1)))
         tf.summary.histogram(name=summary_prefix + "AbsHistMax/" + x.name.replace(":", "_"),
                              values=tf.reduce_max(tf.abs(x), axis=-1))
         mean, variance = tf.nn.moments(x, axes=-1)
         tf.summary.scalar(name=summary_prefix + "Mean/" + x.name.replace(":", "_"),
                           tensor=tf.reduce_mean(mean))
         tf.summary.scalar(name=summary_prefix + "Variance/" + x.name.replace(":", "_"),
                           tensor=tf.reduce_mean(variance))
         tf.summary.histogram(name=summary_prefix + "PosR/" + x.name.replace(":", "_"),
                              values=greater_zero_histogram(x))
       else:
         tf.summary.scalar(name=summary_prefix + "Norm2/" + x.name.replace(":", "_"),
                           tensor=tf.reduce_mean(tf.norm(x, axis=-1)))
         tf.summary.histogram(name=summary_prefix + "AbsHistMax/" + x.name.replace(":", "_"),
                              values=tf.reduce_max(tf.abs(x)))
         mean, variance = tf.nn.moments(x, axes=0)
         tf.summary.scalar(name=summary_prefix + "Mean/" + x.name.replace(":", "_"),
                           tensor=mean[0])
         tf.summary.scalar(name=summary_prefix + "Variance/" + x.name.replace(":", "_"),
                           tensor=variance[0])
       tf.summary.scalar(name=summary_prefix + "PosRatio/" + x.name.replace(":", "_"), tensor=greater_zero_fraction(x))
     except Exception as e:
       logging.warn('Got exception run : %s | %s' % (e, traceback.format_exc()))
       logging.warn("add_dense_output_summary with rank not 2: [%s],shape=[%s]" % (str(x.name), str(x.shape)))


self.main_net = layers.fully_connected(
             self.main_net,
             num_hidden_units,
             utils.getActivationFunctionOp(self.config.activation_op),
             scope=dnn_hidden_layer_scope,
             variables_collections=[self.collections_dnn_hidden_layer],
             outputs_collections=[self.collections_dnn_hidden_output],
             normalizer_fn=layers.batch_norm,
             normalizer_params={"scale": True, "is_training": self.is_training})

base_ops.add_dense_output_summary(self.collections_dnn_hidden_output)

// embedding norm2
def add_embed_layer_norm(layer_tensor, columns):
 if layer_tensor is None:
   return
 i = 0
 for column in sorted(set(columns), key=lambda x: x.key):
   try:
     dim = column.dimension
   except:
     dim = column.embedding_dimension
   tf.summary.scalar(name=column.name, tensor=tf.reduce_mean(tf.norm(layer_tensor[:, i:i + dim], axis=-1)))
   tf.summary.histogram(name="embedding/" + column.name, values=layer_tensor[:, i:i + dim])
   i += dim


# overall fraction
def greater_zero_fraction(value, name=None):
 with tf.name_scope(name, "greater_fraction", [value]):
   value = tf.convert_to_tensor(value, name="value")
   zero = tf.constant(0, dtype=value.dtype, name="zero")
   return math_ops.reduce_mean(
     math_ops.cast(math_ops.greater(value, zero), tf.float32))


# histogram of each sample's zero fraction
def greater_zero_histogram(value, name=None):
 with tf.name_scope(name, "greater_histogram", [value]):
   value = tf.convert_to_tensor(value, name="value")
   zero = tf.constant(0, dtype=value.dtype, name="zero")
   return math_ops.reduce_mean(
     math_ops.cast(math_ops.greater(value, zero), tf.float32), axis=-1)