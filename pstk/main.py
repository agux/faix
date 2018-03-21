from __future__ import print_function
import tensorflow as tf
from model import model, model2, model3, model4, model5, model6
from time import strftime
from data import data2, data4, data5, data6, data7, data8, data9
import os
import numpy as np

EPOCH_SIZE = 444
HIDDEN_SIZE = 256
NUM_LAYERS = 1
MAX_STEP = 60
DROP_OUT = 0.5
LEARNING_RATE = 1e-3
LOG_DIR = 'logdir'

if __name__ == '__main__':
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)

    tf.logging.set_verbosity(tf.logging.INFO)

    print('{} loading test data...'.format(strftime("%H:%M:%S")))
    _, tdata, tlabels, tseqlen = data9.loadTestSet(MAX_STEP)
    print(tdata.shape)
    print(tlabels.shape)
    featSize = tdata.shape[2]
    nclass = tlabels.shape[1]
    data = tf.placeholder(tf.float32, [None, MAX_STEP, featSize], "input")
    target = tf.placeholder(tf.float32, [None, nclass], "labels")
    seqlen = tf.placeholder(tf.int32, [None], "seqlen")
    dropout = tf.placeholder(tf.float32, name="dropout")
    training = tf.placeholder(tf.bool, name="training")
    with tf.Session() as sess:
        model = model6.MRnnPredictorV2(
            data=data, 
            target=target, 
            seqlen=seqlen, 
            num_class=nclass, 
            training=training, 
            dropout=dropout,
            num_hidden=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            learning_rate=LEARNING_RATE)
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(LOG_DIR + "/train", sess.graph)
        test_writer = tf.summary.FileWriter(LOG_DIR + "/test", sess.graph)
        tf.summary.scalar("Loss", model.cost)
        tf.summary.scalar("Accuracy", model.accuracy*100)
        summary = tf.summary.merge_all()
        saver = tf.train.Saver()
        bno = 0
        for epoch in range(EPOCH_SIZE):
            bno = epoch*50
            print('{} running on test set...'.format(strftime("%H:%M:%S")))
            test_summary_str, accuracy = sess.run(
                [summary, model.accuracy], {
                    data: tdata, target: tlabels, seqlen: tseqlen, training: False, dropout: 0})
            print('{} Epoch {:4d} test accuracy {:3.3f}%'.format(
                strftime("%H:%M:%S"), epoch + 1, 100 * accuracy))
            for i in range(50):
                bno = bno+1
                print('{} loading training data for batch {}...'.format(
                    strftime("%H:%M:%S"), bno))
                _, trdata, labels, trseqlen = data9.loadTrainingData(
                    bno, MAX_STEP)
                print('{} training...'.format(strftime("%H:%M:%S")))
                summary_str, _ = sess.run([summary, model.optimize], {
                    data: trdata, target: labels, seqlen: trseqlen, training: True, dropout: DROP_OUT})
                train_writer.add_summary(summary_str, bno)
                test_writer.add_summary(test_summary_str, bno)
                train_writer.flush()
                test_writer.flush()
            checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=bno)
