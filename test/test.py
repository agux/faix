from __future__ import print_function
import tensorflow as tf


def collect_summary(sess, model, base_dir):
    train_writer = tf.compat.v1.summary.FileWriter(base_dir + "/train", sess.graph)
    test_writer = tf.compat.v1.summary.FileWriter(base_dir + "/test", sess.graph)
    with tf.compat.v1.name_scope("Basic"):
        tf.compat.v1.summary.scalar("Loss", model.cost)
        tf.compat.v1.summary.scalar("Accuracy", model.accuracy*100)
    summary = tf.compat.v1.summary.merge_all()
    return summary, train_writer, test_writer