from __future__ import print_function
import tensorflow as tf
k = tf.constant([
    [1, 0, 1],
    [2, 1, 0],
    [0, 0, 1]
], dtype=tf.float32, name='k')
i = tf.constant([
    [4, 3, 1, 0],
    [2, 1, 0, 1],
    [1, 2, 4, 1],
    [3, 1, 0, 2]
], dtype=tf.float32, name='i')
kernel = tf.reshape(k, [3, 3, 1, 1], name='kernel')
image = tf.reshape(i, [1, 4, 4, 1], name='image')

res = tf.squeeze(tf.nn.conv2d(input=image, filters=kernel, strides=[1, 1, 1, 1], padding="VALID"))
# VALID means no padding
with tf.compat.v1.Session() as sess:
   print(sess.run(res))
