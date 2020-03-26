
import tensorflow as tf

d = [{
        'features': tf.float32,
        'seqlens': tf.int32
    }, tf.float32]

print(tf.nest.flatten(d))