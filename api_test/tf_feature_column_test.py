import tensorflow as tf

poi = tf.feature_column.categorical_column_with_hash_bucket("poi", hash_bucket_size=15,
                                                            dtype=tf.dtypes.int64)  # 定义poi特征
poi_ebd = tf.feature_column.embedding_column(poi, dimension=3, combiner="mean")  # poi做embedding
poi_idc = tf.feature_column.indicator_column(poi)  # poi做indicator

feature_column = [poi]
feature_column2 = [poi_ebd]
feature_column3 = [poi_idc]

feature = tf.feature_column.make_parse_example_spec(feature_column)
feature2 = tf.feature_column.make_parse_example_spec(feature_column2)
feature3 = tf.feature_column.make_parse_example_spec(feature_column3)

print(feature)
print(feature2)
print(feature3)
