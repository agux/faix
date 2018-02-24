from __future__ import print_function
import tensorflow as tf
from model import BasicRnnPredictor
from model import BasicCnnPredictor
from model import ShiftRnnPredictor
from model2 import CRnnPredictorV1
from time import strftime
import data4 as dat
import os
import numpy as np

EPOCH_SIZE = 705
HIDDEN_SIZE = 128
NUM_LAYERS = 1
MAX_STEP = 50
LEARNING_RATE = 1e-3
LOG_DIR = 'logdir'

if __name__ == '__main__':
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)

    # tf.logging.set_verbosity(tf.logging.INFO)

    print('{} loading test data...'.format(strftime("%H:%M:%S")))
    _, tdata, tlabels = dat.loadTestSet(MAX_STEP)
    print(tdata.shape)
    print(tlabels.shape)
    featSize = tdata.shape[2]
    numClass = tlabels.shape[1]
    data = tf.placeholder(tf.float32, [None, MAX_STEP, featSize])
    target = tf.placeholder(tf.float32, [None, numClass])
    dropout = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)
    with tf.Session() as sess:
        model = CRnnPredictorV1(
            data, target, dat.TIME_SHIFT+1, training, dropout, num_hidden=HIDDEN_SIZE, num_layers=NUM_LAYERS, learning_rate=LEARNING_RATE)
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
                    data: trdata, target: labels, training: True, dropout: 0.5})
                summary_writer.add_summary(summary_str, bno)
                summary_writer.flush()
                # print('{} tagging data as trained, batch no: {}'.format(
                #     strftime("%H:%M:%S"), bno))
                # dat.tagDataTrained(uuid, bno)
            print('{} running on test set...'.format(strftime("%H:%M:%S")))
            accuracy = sess.run(
                model.accuracy, {data: tdata, target: tlabels, training: False, dropout: 0})
            print('{} Epoch {:4d} test accuracy {:3.3f}%'.format(
                strftime("%H:%M:%S"), epoch + 1, 100 * accuracy))
            checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=bno)
