from __future__ import print_function
import tensorflow as tf
from model import model, model2, model3, model4, model5, model6
from time import strftime
from data import data2, data4, data5, data6, data7, data8, data9
import os
import numpy as np

EPOCH_SIZE = 443
HIDDEN_SIZE = 256
NUM_LAYERS = 3
MAX_STEP = 60
DROP_OUT = 0.7
LEARNING_RATE = 1e-3
LOG_DIR = 'logdir'


def collect_summary(sess, model):
    train_writer = tf.summary.FileWriter(LOG_DIR + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + "/test", sess.graph)
    with tf.name_scope("Summary"):
        tf.summary.scalar("Loss", model.cost)
        tf.summary.scalar("Accuracy", model.accuracy*100)
    summary = tf.summary.merge_all()
    return summary, train_writer, test_writer


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
    classes = [i-nclass//2 for i in range(nclass)]
    data = tf.placeholder(tf.float32, [None, MAX_STEP, featSize], "input")
    target = tf.placeholder(tf.float32, [None, nclass], "labels")
    seqlen = tf.placeholder(tf.int32, [None], "seqlen")
    dropout = tf.placeholder(tf.float32, name="dropout")
    training = tf.placeholder(tf.bool, name="training")
    with tf.Session() as sess:
        model = model6.MRnnPredictorV3(
            data=data,
            target=target,
            seqlen=seqlen,
            classes=classes,
            training=training,
            dropout=dropout,
            num_hidden=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            learning_rate=LEARNING_RATE)
        # Isolate the variables stored behind the scenes by the metric operation
        metric_local_vars = tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES, scope="Precisions") + tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES, scope="Recalls")
        metric_vars_initializer = tf.variables_initializer(
            var_list=metric_local_vars)
        sess.run(tf.group(tf.global_variables_initializer(),
                          metric_vars_initializer))
        summary, train_writer, test_writer = collect_summary(sess, model)
        saver = tf.train.Saver()
        bno = 0
        for epoch in range(EPOCH_SIZE):
            bno = epoch*50
            print('{} running on test set...'.format(strftime("%H:%M:%S")))
            feeds = {data: tdata, target: tlabels,
                     seqlen: tseqlen, training: False, dropout: 0}
            accuracy, test_summary_str = sess.run(
                [model.accuracy, summary, model.precisions, model.recalls], feeds)[:2]
            print('{} Epoch {:4d} test accuracy {:3.3f}%'.format(
                strftime("%H:%M:%S"), epoch + 1, 100 * accuracy))
            for i in range(50):
                sess.run(metric_vars_initializer)
                bno = bno+1
                print('{} loading training data for batch {}...'.format(
                    strftime("%H:%M:%S"), bno))
                _, trdata, labels, trseqlen = data9.loadTrainingData(
                    bno, MAX_STEP)
                print('{} training...'.format(strftime("%H:%M:%S")))
                feeds = {data: trdata, target: labels,
                         seqlen: trseqlen, training: True, dropout: DROP_OUT}
                summary_str = sess.run(
                    [summary, model.optimize, model.precisions, model.recalls], feeds)[0]
                train_writer.add_summary(summary_str, bno)
                test_writer.add_summary(test_summary_str, bno)
                train_writer.flush()
                test_writer.flush()
            checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=bno)
            sess.run(metric_vars_initializer)
