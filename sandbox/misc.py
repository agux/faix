import tensorflow as tf

EPSILON = 1e-6


i = tf.keras.Input(shape=[2, 3])
s = tf.shape(i)
f = tf.fill([s[0], 4,5], EPSILON, name="f")
f2 = tf.fill([s[0], 2], EPSILON, name="f2")
# c = tf.constant(EPSILON, shape=[s[0], 2, 2])
# c = tf.constant_initializer(EPSILON)(shape=[None, 2,2])
# z = tf.zeros([s[0], 2], name="aaa")
# z2 = tf.

# print("None: {}".format(s[0]))
# print("fill: {}".format(f))
# print("constant: {}".format(c))
print(f)
print(f2)