from __future__ import print_function
import tensorflow as tf
from time import strftime
import numpy as np

t1 = [1, 0, 0, 0]
t2 = [
    [1.2, 3.1, 0.7, 2.331],
    [-1., -2., -3., 0.000]
]
t3 = [4, 5, 6]

labels = np.asarray([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

logits = np.asarray([
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
    [0., 0., 1., 0.],  # W
    [0., 1., 0., 0.],
    [1., 0., 0., 0.],
    [1., 0., 0., 0.]  # W
])

# mask = [
#     [1, 0, 0, 0]
# ]

mask1 = np.zeros((1, logits.shape[1]), int)
mask1[0][0] = 1

mask2 = np.zeros((1, logits.shape[1]), int)
mask2[0][3] = 1


def model(input):
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    print("update_ops:{}".format(update_ops))
    with tf.control_dependencies(update_ops):
        return tf.one_hot(tf.argmax(input=input, axis=1), 4, axis=-1)


with tf.compat.v1.Session() as sess:
    input = tf.compat.v1.placeholder(tf.float32, [None, 4])
    # print(sess.run(tf.cast(t1, tf.bool)))
    # print(sess.run(tf.argmax(t2, 1)))
    onehot = tf.one_hot(tf.argmax(input=t2, axis=1), 4, axis=-1)
    print(sess.run(onehot))
    print(sess.run(tf.cast(onehot, tf.bool)))
    # tf.one_hot(tf.argmax(self.prediction, 1), size, axis = -1),
    # print([3, 3, 3]+t1+t3)

    r1, _ = tf.compat.v1.metrics.recall(
        labels=labels,
        predictions=model(input),
        weights=mask1,
        updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS)
    p1, _ = tf.compat.v1.metrics.precision(
        labels=labels,
        predictions=model(input),
        weights=mask1,
        updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS)
    r2, _ = tf.compat.v1.metrics.recall(
        labels=labels,
        predictions=model(input),
        weights=mask2,
        updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS)
    p2, _ = tf.compat.v1.metrics.precision(
        labels=labels,
        predictions=model(input),
        weights=mask2,
        updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS)
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    # sess.run([r_op, p_op])

    sess.run(model(input), feed_dict={input: logits})

    print("recall:{}".format(sess.run(r1)))
    print("precision:{}".format(sess.run(p1)))
    print("recall:{}".format(sess.run(r2)))
    print("precision:{}".format(sess.run(p2)))
