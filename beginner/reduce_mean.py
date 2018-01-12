import tensorflow as tf

i = tf.constant([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
], dtype=tf.float32, name='i')

res = tf.reduce_mean(i)

with tf.Session() as sess:
   print sess.run(res)
