from __future__ import print_function
import tensorflow as tf
from time import strftime
import numpy as np

print(5/2.0)

seq = [
    [[1, 2, 3], [2, 3, 4]],
    [[3, 4, 5], [1, 0, 0]]
]

print("{}".format(np.concatenate(seq,1)))


print(5//2)

flag = "TRN_{}_{}".format(strftime("%Y%m%d_%H%M%S"), 3)
print(flag)


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length



le = length(seq)

ts = tf.zeros([2, 2], dtype=tf.float32)

t = tf.constant([[[1, 1, 1], [2, 2, 2], [7, 7, 7], [8, 8, 8]],
                 [[3, 3, 3], [4, 4, 4], [6, 6, 6], [0, 0, 0]],
                 [[5, 5, 5], [9, 9, 9], [0, 0, 0], [0, 0, 0]]])

def f(p1,p2,p3,p4,ph):
    print("received p3:{}".format(p3))
    return "{}+{}".format(p1, p2), p2

with tf.Session() as sess:
    # l = sess.run(le)
    # print(l)
    # print(ts)
    # print([1, 2] + [2, 3] + [4])
    st = tf.gather_nd(
        t, [[[0, 2], [0, 3]], [[1, 1], [1, 2]], [[2, 0], [2, 1]]])
    print(st.get_shape())
    print(st.eval())
    print(t.get_shape())
    px1 = tf.placeholder(tf.string, [2, None, 1])
    x1 = np.array([[['a'], ['b'], ['c']],
                   [['1'], ['2'], ['3']]])
    x2 = np.array([5, 6])
    ph = tf.placeholder(tf.float32)

    # input = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
    elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
    r = tf.map_fn(lambda x: f(x[0],x[1],'3','4',ph), (px1, x2))
    print(sess.run(r, feed_dict={px1: x1, ph: 0.5}))
    # a = tf.constant([[1, 2, 3], [4, 5, 6]])
    # b = tf.constant([True, False], dtype=tf.bool)

    # c = tf.map_fn(lambda x: (x[0], x[1]), (a, b))
    # print(c)

