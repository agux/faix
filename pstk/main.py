from __future__ import print_function
import tensorflow as tf
from model import BasicSecGradePredictor
from time import strftime
import data2 as dat
import os
import numpy as np

EPOCH_SIZE = 705
HIDDEN_SIZE = 64
NUM_LAYERS = 1
#BATCH_SIZE = 200
W_SIZE = 10
MAX_STEP = 10
LEARNING_RATE = 1e-3
LOG_DIR = 'logdir'

if __name__ == '__main__':
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    print('{} loading test data...'.format(strftime("%H:%M:%S")))
    _, tdata, tlabels = dat.loadTestSet(MAX_STEP)
    featSize = np.array(tdata).shape[2]
    numClass = np.array(tlabels).shape[1]
    data = tf.placeholder(tf.float32, [None, MAX_STEP, featSize])
    target = tf.placeholder(tf.float32, [None, numClass])
    training = tf.placeholder(tf.bool)
    model = BasicSecGradePredictor(
        data, target, W_SIZE, training, num_hidden=HIDDEN_SIZE, num_layers=NUM_LAYERS, learning_rate=LEARNING_RATE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        tf.summary.scalar("Train Loss", model.cost)
        tf.summary.scalar("Train Accuracy", model.accuracy*100)
        summary = tf.summary.merge_all()
        saver = tf.train.Saver()
        bno = 0
        for epoch in range(EPOCH_SIZE):
            bno = epoch*50
            for i in range(50):
                bno = bno+1
                print('{} loading training data for batch {}...'.format(
                    strftime("%H:%M:%S"), bno))
                uuid, trdata, labels = dat.loadTrainingData(bno, MAX_STEP)
                print('{} training...'.format(strftime("%H:%M:%S")))
                summary_str, _ = sess.run([summary, model.optimize], {
                    data: trdata, target: labels, training: True})
                summary_writer.add_summary(summary_str, bno)
                summary_writer.flush()
                # print('{} tagging data as trained, batch no: {}'.format(
                #     strftime("%H:%M:%S"), bno))
                # dat.tagDataTrained(uuid, bno)
            print('{} running on test set...'.format(strftime("%H:%M:%S")))
            accuracy = sess.run(
                model.accuracy, {data: tdata, target: tlabels, training: False})
            print('{} Epoch {:4d} test accuracy {:3.3f}%'.format(
                strftime("%H:%M:%S"), epoch + 1, 100 * accuracy))
            checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=bno)
