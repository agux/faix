import tensorflow as tf

e = tf.get_variable("embedding", [5, 10], tf.float32)

i = tf.nn.embedding_lookup(e, [0, 1, 3, 2, 4])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(i))
    print(i[:, 1, :])
