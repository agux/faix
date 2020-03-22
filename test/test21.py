from __future__ import print_function
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import tensorflow as tf
from pstk.model import model4
from time import strftime
from pstk.data import data as data0
from pstk.data import data14
from test import collect_summary
import os
import numpy as np
import math

EPOCH_SIZE = 444
RNN_LAYERS = 1
LAYER_WIDTH = 256
MAX_STEP = 50
TIME_SHIFT = 9
DROP_OUT = 0.5
LEARNING_RATE = 1e-3
LOG_DIR = 'logdir'


def run():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    loader = data14.DataLoader(TIME_SHIFT)
    print('{} loading test data...'.format(strftime("%H:%M:%S")))
    tuuids, tdata, tlabels, tseqlen = loader.loadTestSet(MAX_STEP)
    print(tdata.shape)
    print(tlabels.shape)
    featSize = tdata.shape[2]
    nclass = tlabels.shape[1]
    classes = [i-nclass//2 for i in range(nclass)]
    data = tf.compat.v1.placeholder(tf.float32, [None, MAX_STEP, featSize], "input")
    target = tf.compat.v1.placeholder(tf.float32, [None, nclass], "labels")
    seqlen = tf.compat.v1.placeholder(tf.int32, [None], "seqlen")
    dropout = tf.compat.v1.placeholder(tf.float32, [], name="dropout")
    training = tf.compat.v1.placeholder(tf.bool, [], name="training")
    with tf.compat.v1.Session() as sess:
        model = model4.ERnnPredictorV3(
            data=data,
            target=target,
            seqlen=seqlen,
            classes=classes,
            num_layers=RNN_LAYERS,
            num_hidden=LAYER_WIDTH,
            dropout=dropout,
            training=training,
            learning_rate=LEARNING_RATE)
        stime = '{}'.format(strftime("%Y-%m-%d %H:%M:%S"))
        model_name = model.getName()
        f = __file__
        fbase = f[f.rfind('/')+1:f.rindex('.py')]
        base_dir = '{}/{}_{}/{}'.format(LOG_DIR, fbase,
                                        model_name, strftime("%Y%m%d_%H%M%S"))
        print('{} using model: {}'.format(strftime("%H:%M:%S"), model_name))
        if tf.io.gfile.exists(base_dir):
            tf.io.gfile.rmtree(base_dir)
        tf.io.gfile.makedirs(base_dir)
        # Isolate the variables stored behind the scenes by the metric operation
        metric_local_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope="Precisions") + tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope="Recalls")
        metric_vars_initializer = tf.compat.v1.variables_initializer(
            var_list=metric_local_vars)
        sess.run(tf.group(tf.compat.v1.global_variables_initializer(),
                          metric_vars_initializer))
        summary, train_writer, test_writer = collect_summary(
            sess, model, base_dir)
        saver = tf.compat.v1.train.Saver()
        bno = 0
        for epoch in range(EPOCH_SIZE):
            bno = epoch*50
            print('{} running on test set...'.format(strftime("%H:%M:%S")))
            feeds = {data: tdata, target: tlabels,
                     seqlen: tseqlen, dropout: 0, training: False}
            accuracy, worst, test_summary_str = sess.run(
                [model.accuracy, model.worst, summary, model.precisions[1], model.recalls[1], model.f_score], feeds)[:3]
            bidx, max_entropy, predict, actual = worst[0], worst[1], worst[2], worst[3]
            print('{} Epoch {} test accuracy {:3.3f}% max_entropy {:3.4f} predict {} actual {} uuid {}'.format(
                strftime("%H:%M:%S"), epoch, 100. * accuracy, max_entropy, predict, actual, tuuids[bidx]))
            data0.save_worst_rec(model_name, stime, "test", epoch,
                                 tuuids[bidx], max_entropy, predict, actual)
            summary_str = None
            for i in range(50):
                sess.run(metric_vars_initializer)
                bno = bno+1
                print('{} loading training data for batch {}...'.format(
                    strftime("%H:%M:%S"), bno))
                truuids, trdata, labels, trseqlen = loader.loadTrainingData(
                    bno, MAX_STEP)
                print('{} training...'.format(strftime("%H:%M:%S")))
                feeds = {data: trdata, target: labels,
                         seqlen: trseqlen, dropout: DROP_OUT, training: True}
                summary_str, worst = sess.run(
                    [summary, model.worst, model.optimize, model.precisions[1], model.recalls[1], model.f_score], feeds)[:2]
                bidx, max_entropy, predict, actual = worst[0], worst[1], worst[2], worst[3]
                print('{} bno {} max_entropy {:3.4f} predict {} actual {}'.format(
                    strftime("%H:%M:%S"), bno, max_entropy, predict, actual))
                data0.save_worst_rec(model_name, stime, "train", bno,
                                     truuids[bidx], max_entropy, predict, actual)
                train_writer.add_summary(summary_str, bno)
                test_writer.add_summary(test_summary_str, bno)
                train_writer.flush()
                test_writer.flush()
            checkpoint_file = os.path.join(base_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=bno)
            sess.run(metric_vars_initializer)
        # test last epoch
        print('{} running on test set...'.format(strftime("%H:%M:%S")))
        feeds = {data: tdata, target: tlabels, seqlen: tseqlen,
                 dropout: 0, training: False}
        accuracy, worst, test_summary_str = sess.run(
            [model.accuracy, model.worst, summary, model.precisions[1], model.recalls[1], model.f_score], feeds)[:3]
        bidx, max_entropy, predict, actual = worst[0], worst[1], worst[2], worst[3]
        print('{} Epoch {} test accuracy {:3.3f}% max_entropy {:3.4f} predict {} actual {}'.format(
            strftime("%H:%M:%S"), EPOCH_SIZE, 100. * accuracy, max_entropy, predict, actual))
        data0.save_worst_rec(model_name, stime, "test", EPOCH_SIZE,
                             tuuids[bidx], max_entropy, predict, actual)
        train_writer.add_summary(summary_str, bno)
        test_writer.add_summary(test_summary_str, bno)
        train_writer.flush()
        test_writer.flush()


if __name__ == '__main__':
    run()
