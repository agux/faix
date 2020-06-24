import tensorflow as tf
import numpy as np


def reduce_prod(x, axis, name=None):
    '''
    Uses tf.cumprod and tf.gather_nd as a workaround to the poor performance of calculating tf.reduce_prod's gradient
    on CPU.
    '''
    with tf.name_scope(name or "c_reduce_prod"):
        cp=tf.math.cumprod(x, axis, reverse=True)
        shape = cp.shape.as_list()
        r = len(shape)
        begin = np.zeros([r], np.int)
        size = [-1 if e is None else e for e in shape]
        size[-1] = 1
        sliced = tf.slice(cp, begin, size)
        return tf.squeeze(sliced, -1)
        # size=tf.shape(cp)[0]
        # idx1=tf.range(tf.cast(size, tf.float32), dtype=tf.float32)
        # idx2=tf.zeros([size], tf.float32)
        # indices = tf.stack([idx1, idx2], 1)
        # return tf.gather_nd(cp, tf.cast(indices, tf.int32))
        

x = tf.constant([[[1,2,3],[2,3,4]],
                [[2,3,4],[3,4,5]]])

print(x)

reduced = tf.math.reduce_prod(x, 2)

print(reduced)

reduced = reduce_prod(x, 2)

print(reduced)