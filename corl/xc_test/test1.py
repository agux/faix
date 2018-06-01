from __future__ import print_function
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import tensorflow as tf
# pylint: disable-msg=E0401
from model import base as model0
from xc_data import base as data0
from time import strftime
import os
import numpy as np
import math

N_TEST = 100
TEST_INTERVAL = 50
EPOCH_SIZE = 94315 // TEST_INTERVAL
LAYER_WIDTH = 256
MAX_STEP = 30
TIME_SHIFT = 0
LEARNING_RATE = 1e-3
USE_PEEPHOLES = True
TIED = False
LOG_DIR = 'logdir'

# pylint: disable-msg=E0601


def collect_summary(sess, model, base_dir):
    train_writer = tf.summary.FileWriter(base_dir + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(base_dir + "/test", sess.graph)
    with tf.name_scope("Basic"):
        tf.summary.scalar("Mean_Diff", tf.sqrt(model.cost))
    summary = tf.summary.merge_all()
    return summary, train_writer, test_writer


def run():
    tf.logging.set_verbosity(tf.logging.INFO)
    loader = data0.DataLoader(TIME_SHIFT)
    print('{} loading test data...'.format(strftime("%H:%M:%S")))
    tuuids, tdata, txcorls, tseqlen = loader.loadTestSet(MAX_STEP, N_TEST)
    print('input shape: {}'.format(tdata.shape))
    print('target shape: {}'.format(txcorls.shape))
    featSize = tdata.shape[2]
    data = tf.placeholder(tf.float32, [None, MAX_STEP, featSize], "input")
    target = tf.placeholder(tf.float32, [None], "target")
    seqlen = tf.placeholder(tf.int32, [None], "seqlen")
    with tf.Session() as sess:
        model = model0.SRnnRegressor(
            data=data,
            target=target,
            seqlen=seqlen,
            cell='grid3lstm',
            use_peepholes=USE_PEEPHOLES,
            tied=TIED,
            layer_width=LAYER_WIDTH,
            learning_rate=LEARNING_RATE)
        model_name = model.getName()
        f = __file__
        fbase = f[f.rfind('/')+1:f.rindex('.py')]
        base_dir = '{}/{}_{}/{}'.format(LOG_DIR, fbase,
                                        model_name, strftime("%Y%m%d_%H%M%S"))
        print('{} using model: {}'.format(strftime("%H:%M:%S"), model_name))
        if tf.gfile.Exists(base_dir):
            tf.gfile.DeleteRecursively(base_dir)
        tf.gfile.MakeDirs(base_dir)
        sess.run(tf.global_variables_initializer())
        summary, train_writer, test_writer = collect_summary(
            sess, model, base_dir)
        saver = tf.train.Saver()
        bno = 0
        for epoch in range(EPOCH_SIZE):
            bno = epoch*TEST_INTERVAL
            print('{} running on test set...'.format(strftime("%H:%M:%S")))
            feeds = {data: tdata, target: txcorls, seqlen: tseqlen}
            mse, worst, test_summary_str = sess.run(
                [model.cost, model.worst, summary], feeds)
            bidx, max_diff, predict, actual = worst[0], worst[1], worst[2], worst[3]
            print('{} Epoch {} mse {:3.5f} max_diff {:3.4f} predict {} actual {} uuid {}'.format(
                strftime("%H:%M:%S"), epoch, mse, max_diff, predict, actual, tuuids[bidx]))
            summary_str = None
            for _ in range(TEST_INTERVAL):
                bno = bno+1
                print('{} loading training data for batch {}...'.format(
                    strftime("%H:%M:%S"), bno))
                _, trdata, trxcorls, trseqlen = loader.loadTrainingData(
                    bno, MAX_STEP)
                print('{} training...'.format(strftime("%H:%M:%S")))
                feeds = {data: trdata, target: trxcorls,  seqlen: trseqlen}
                summary_str, worst = sess.run(
                    [summary, model.worst, model.optimize], feeds)[:-1]
                bidx, max_diff, predict, actual = worst[0], worst[1], worst[2], worst[3]
                print('{} bno {} max_diff {:3.4f} predict {} actual {}'.format(
                    strftime("%H:%M:%S"), bno, max_diff, predict, actual))
                train_writer.add_summary(summary_str, bno)
                test_writer.add_summary(test_summary_str, bno)
                train_writer.flush()
                test_writer.flush()
            checkpoint_file = os.path.join(base_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=bno)
        # test last epoch
        print('{} running on test set...'.format(strftime("%H:%M:%S")))
        feeds = {data: tdata, target: txcorls, seqlen: tseqlen}
        mse, worst, test_summary_str = sess.run(
            [model.cost, model.worst, summary], feeds)
        bidx, max_diff, predict, actual = worst[0], worst[1], worst[2], worst[3]
        print('{} Epoch {} mse {:3.5f} max_diff {:3.4f} predict {} actual {} uuid {}'.format(
            strftime("%H:%M:%S"), epoch, mse, max_diff, predict, actual, tuuids[bidx]))
        train_writer.add_summary(summary_str, bno)
        test_writer.add_summary(test_summary_str, bno)
        train_writer.flush()
        test_writer.flush()


if __name__ == '__main__':
    run()
