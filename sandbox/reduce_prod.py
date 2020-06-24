import tensorflow as tf


def reduce_prod(x, axis, name=None):
    '''
    Uses tf.cumprod and tf.gather_nd as a workaround to the poor performance of calculating tf.reduce_prod's gradient
    on CPU.
    '''
    with tf.name_scope(name or "c_reduce_prod"):
        cp=tf.math.cumprod(x, axis, reverse=True)
        shape = tf.shape(cp)
        r = tf.size(shape)
        begin = tf.zeros([r], tf.int32)
        size = tf.tensor_scatter_nd_update(shape, [[r-1]], [1])
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