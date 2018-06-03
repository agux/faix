from __future__ import print_function
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import tensorflow as tf
# pylint: disable-msg=E0401
from model import drnn_regressor
from wc_data import base as data0
from test1 import collect_summary
from time import strftime
import os
import numpy as np
import math

N_TEST = 100
TEST_INTERVAL = 50
LAYER_WIDTH = 512
MAX_STEP = 50
TIME_SHIFT = 4
LEARNING_RATE = 1e-3
USE_PEEPHOLES = True
TIED = False
LOG_DIR = 'logdir'

# pylint: disable-msg=E0601


k_cols = [
    "lr", "lr_h", "lr_o", "lr_l",
    "lr_h_c", "lr_o_c", "lr_l_c",
    "lr_ma5", "lr_ma5_h", "lr_ma5_o", "lr_ma5_l",
    "lr_ma10", "lr_ma10_h", "lr_ma10_o", "lr_ma10_l",
    "lr_vol", "lr_vol5", "lr_vol10"
]


def run():
    tf.logging.set_verbosity(tf.logging.INFO)
    loader = data0.DataLoader(TIME_SHIFT, k_cols)
    print('{} loading test data...'.format(strftime("%H:%M:%S")))
    tuuids, tdata, tvals, tseqlen = loader.loadTestSet(MAX_STEP, N_TEST)
    print('input shape: {}'.format(tdata.shape))
    print('target shape: {}'.format(tvals.shape))
    featSize = tdata.shape[2]
    data = tf.placeholder(tf.float32, [None, MAX_STEP, featSize], "input")
    target = tf.placeholder(tf.float32, [None], "target")
    seqlen = tf.placeholder(tf.int32, [None], "seqlen")
    with tf.Session() as sess:
        model = drnn_regressor.DRnnRegressorV1(
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
        epoch = 0
        while True:
            bno = epoch*TEST_INTERVAL
            print('{} running on test set...'.format(strftime("%H:%M:%S")))
            feeds = {data: tdata, target: tvals, seqlen: tseqlen}
            mse, worst, test_summary_str = sess.run(
                [model.cost, model.worst, summary], feeds)
            bidx, max_diff, predict, actual = worst[0], worst[1], worst[2], worst[3]
            print('{} Epoch {} diff {:3.5f} max_diff {:3.4f} predict {} actual {} uuid {}'.format(
                strftime("%H:%M:%S"), epoch, math.sqrt(mse), max_diff, predict, actual, tuuids[bidx]))
            summary_str = None
            fin = False
            for _ in range(TEST_INTERVAL):
                bno = bno+1
                print('{} loading training data for batch {}...'.format(
                    strftime("%H:%M:%S"), bno))
                _, trdata, trvals, trseqlen = loader.loadTrainingData(
                    bno, MAX_STEP)
                if len(trdata) > 0:
                    print('{} training...'.format(strftime("%H:%M:%S")))
                else:
                    print('{} end of training data, finish training.'.format(
                        strftime("%H:%M:%S")))
                    fin = True
                    break
                feeds = {data: trdata, target: trvals,  seqlen: trseqlen}
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
            epoch += 1
            if fin:
                break
        # test last epoch
        print('{} running on test set...'.format(strftime("%H:%M:%S")))
        feeds = {data: tdata, target: tvals, seqlen: tseqlen}
        mse, worst, test_summary_str = sess.run(
            [model.cost, model.worst, summary], feeds)
        bidx, max_diff, predict, actual = worst[0], worst[1], worst[2], worst[3]
        print('{} Epoch {} diff {:3.5f} max_diff {:3.4f} predict {} actual {} uuid {}'.format(
            strftime("%H:%M:%S"), epoch, math.sqrt(mse), max_diff, predict, actual, tuuids[bidx]))
        train_writer.add_summary(summary_str, bno)
        test_writer.add_summary(test_summary_str, bno)
        train_writer.flush()
        test_writer.flush()


if __name__ == '__main__':
    run()
