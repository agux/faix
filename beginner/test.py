import tensorflow as tf

e = tf.get_variable("onehot", [1, 1], tf.int32)

i = tf.one_hot(e, 3)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(i))
    print(i)
