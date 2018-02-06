from __future__ import print_function
import tensorflow as tf
from time import strftime

flag = "TRN_{}_{}".format(strftime("%Y%m%d_%H%M%S"), 3)
print(flag)

def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

seq = [
  [[1,2,3],[2,3,4]],
  [[3,4,5],[1,0,0]]
]
le = length(seq)

ts = tf.zeros([2,2], dtype=tf.float32)


with tf.Session() as sess:
    l = sess.run(le)
    print(l)
    print(ts)


