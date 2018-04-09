from __future__ import print_function
import tensorflow as tf


def collect_summary(sess, model, base_dir):
    train_writer = tf.summary.FileWriter(base_dir + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(base_dir + "/test", sess.graph)
    with tf.name_scope("Basic"):
        tf.summary.scalar("Loss", model.cost)
        tf.summary.scalar("Accuracy", model.accuracy*100)
    summary = tf.summary.merge_all()
    return summary, train_writer, test_writer