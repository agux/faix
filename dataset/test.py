

import tensorflow as tf
import numpy as np


global_value = None


def parse_fn(bno):
    global global_value
    print("parsing bno {}, global value: {}".format(bno, global_value))
    data = np.random.sample((3, 2, 1))
    label = np.random.sample((1))
    if global_value is None:
        global_value = np.random.sample(1)
    return data, label


def foo_parse_fn(foo):
    print("parsing foo_parse_fn")
    foo.say()
    data = np.random.sample((3, 2, 1))
    label = np.random.sample((1))
    return data, label


EPOCHS = 11
# create a random vector of shape (100,2)
train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
test = np.array([99])
# make a dataset from a numpy array


class foo:
    def __init__(self, bar):
        self._bar = bar

    @staticmethod
    def say(self):
        print("saying {} from foo".format(self._bar))


with tf.compat.v1.Session() as sess:
    # foos = [foo(i+10) for i in range(EPOCHS)]
    # train_dataset = tf.data.Dataset.from_tensor_slices(
    #     foos).map(lambda f: tuple(tf.py_func(foo_parse_fn, [f], [tf.float64, tf.float64]))).batch(1).prefetch(2)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        train).map(lambda f: tuple(tf.compat.v1.py_func(parse_fn, [f], [tf.float64, tf.float64]))).batch(1).prefetch(2)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        test).map(lambda f: tuple(tf.compat.v1.py_func(parse_fn, [f], [tf.float64, tf.float64]))).batch(1).repeat()

    train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
    test_iterator = tf.compat.v1.data.make_one_shot_iterator(test_dataset)

    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    iter = tf.compat.v1.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    next_el = iter.get_next()
    next_el = tf.tuple(tensors=[tf.squeeze(next_el[0], [0]),
                        tf.squeeze(next_el[1], [0])])

    train_handle, test_handle = sess.run(
        [train_iterator.string_handle(), test_iterator.string_handle()])
    # initialize the iterator
    # sess.run([test_iterator.initializer])

    # simulate training
    for i in range(EPOCHS):
        if i % 3 == 0:
            # run validation
            out = sess.run(next_el, feed_dict={handle: test_handle})
            print("test out: {}".format(out))
        try:
            out = sess.run(next_el, feed_dict={handle: train_handle})
            print("iter {} train out: {}".format(i, out))
        except tf.errors.OutOfRangeError:
            print("End of Dataset.")
            break
    out = sess.run(next_el, feed_dict={handle: test_handle})
    print("test out: {}".format(out))
