from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
print("downloading mnist data...")
mnist = input_data.read_data_sets(
    "mnist/input_data", source_url="http://yann.lecun.com/exdb/mnist/", one_hot=True)

import tensorflow as tf
from model import SecurityGradePredictor
from time import strftime
import data as dat
import os

EPOCH_SIZE = 5000
HIDDEN_SIZE = 200
NUM_LAYERS = 5
BATCH_SIZE = 1000
W_SIZE = NUM_LAYERS
MAX_STEP = 28
FEAT_SIZE = 28
NUM_CLASSES = 10
LEARNING_RATE = 1e-3
LOG_DIR = 'logdir'

if __name__ == '__main__':
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)

    data = tf.placeholder(tf.float32, [None, MAX_STEP, FEAT_SIZE])
    target = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    training = tf.placeholder(tf.bool)
    model = SecurityGradePredictor(
        data, target, W_SIZE, training, num_hidden=HIDDEN_SIZE, num_layers=NUM_LAYERS, learning_rate=LEARNING_RATE)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())
        tf.summary.scalar("Train Loss", model.cost)
        tf.summary.scalar("Train Accuracy", model.accuracy*100)
        summary = tf.summary.merge_all()
        bno = 0
        for epoch in range(EPOCH_SIZE):
            bno = epoch*50
            for i in range(50):
                bno = bno+1
                print('{} loading training data for batch {}...'.format(
                    strftime("%H:%M:%S"), bno))
                batch_x, batch_y = mnist.train.next_batch(
                    batch_size=BATCH_SIZE)
                batch_x = batch_x.reshape((BATCH_SIZE, MAX_STEP, FEAT_SIZE))
                print('{} training...'.format(strftime("%H:%M:%S")))
                summary_str, _ = sess.run([summary, model.optimize], {
                    data: batch_x, target: batch_y, training: True})
                summary_writer.add_summary(summary_str, bno)
                summary_writer.flush()
                # print('{} tagging data as trained, batch no: {}'.format(
                #     strftime("%H:%M:%S"), bno))
                # dat.tagDataTrained(uuid, bno)
            print('{} running on test set...'.format(strftime("%H:%M:%S")))
            test_data = mnist.test.images.reshape((-1, MAX_STEP, FEAT_SIZE))
            test_label = mnist.test.labels
            accuracy = sess.run(
                model.accuracy, {data: test_data, target: test_label, training: False})
            print('{} Epoch {:4d} test accuracy {:3.3f}%'.format(
                strftime("%H:%M:%S"), epoch + 1, 100 * accuracy))
            checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_file,
                       global_step=bno)
