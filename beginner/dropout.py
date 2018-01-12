import tensorflow as tf

i = tf.constant([
    [-1, 3, 1, -0.0001],
    [2, 1, 0, 1],
    [1, 2, -4, 1],
    [-9, 1, 0, 2]
], dtype=tf.float32, name='i')

# b = tf.constant([ 5, 4, 3, 2], dtype=tf.float32)

res = tf.nn.dropout(i, 0.8)

with tf.Session() as sess:
   print sess.run(res)
