import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


# init tensor
# x = tf.constant(4, shape=(1,1), dtype=tf.float32)
# x = tf.constant([[1,2,3], [4,5,6]])
# x = tf.ones((3,3))
# x = tf.random.normal((3, 3), mean=0, stddev=1)
# x = tf.random.uniform((1,3), minval=0, maxval=1)
# x = tf.range(9)
# print(x)

# mathematical operators
# x = tf.constant([1,2,3])
# y = tf.constant([9,8,7])

# z = tf.add(x, y)
# z = tf.tensordot(x, y, axes=1)
# print(z)

# Indexing
x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
print(x[3:])