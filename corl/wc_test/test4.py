from __future__ import print_function
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import argparse
import tensorflow as tf
# pylint: disable-msg=E0401
from model import base as base_model
from wc_data import base as data0
from test1 import collect_summary
from time import strftime
import os
import numpy as np
import math
import multiprocessing
import shutil

N_TEST = 100
TEST_INTERVAL = 50
SAVE_INTERVAL = 10
LAYER_WIDTH = 512
MAX_STEP = 50
TIME_SHIFT = 4
DIM = 5
LEARNING_RATE = 1e-3
LOG_DIR = 'logdir'

# pylint: disable-msg=E0601


k_cols = [
    "lr", "lr_h", "lr_o", "lr_l",
    "lr_h_c", "lr_o_c", "lr_l_c",
    "lr_ma5", "lr_ma5_h", "lr_ma5_o", "lr_ma5_l",
    "lr_ma10", "lr_ma10_h", "lr_ma10_o", "lr_ma10_l",
    "lr_vol", "lr_vol5", "lr_vol10"
]

parser = argparse.ArgumentParser()
parser.add_argument('parallel', type=int, nargs='?', help='database operation parallel level',
                    default=multiprocessing.cpu_count())
parser.add_argument(
    '--restart', help='restart training', action='store_true')
args = parser.parse_args()


def run():
    global args
    tf.logging.set_verbosity(tf.logging.INFO)
    loader = data0.DataLoader(TIME_SHIFT, k_cols, args.parallel)
    print('{} loading test data...'.format(strftime("%H:%M:%S")))
    tuuids, tdata, tvals, tseqlen = loader.loadTestSet(MAX_STEP, N_TEST)
    print('input shape: {}'.format(tdata.shape))
    print('target shape: {}'.format(tvals.shape))
    featSize = tdata.shape[2]
    data = tf.placeholder(tf.float32, [None, MAX_STEP, featSize], "input")
    target = tf.placeholder(tf.float32, [None], "target")
    seqlen = tf.placeholder(tf.int32, [None], "seqlen")
    with tf.Session() as sess:
        model = base_model.SRnnRegressorV3(
            data=data,
            target=target,
            seqlen=seqlen,
            dim=DIM,
            layer_width=LAYER_WIDTH,
            learning_rate=LEARNING_RATE)
        model_name = model.getName()
        print('{} using model: {}'.format(strftime("%H:%M:%S"), model_name))
        f = __file__
        testn = f[f.rfind('/')+1:f.rindex('.py')]
        base_dir = '{}/{}_{}'.format(LOG_DIR, testn, model_name)
        training_dir = os.path.join(base_dir, 'training')
        summary_dir = os.path.join(training_dir, 'summary')
        checkpoint_file = os.path.join(training_dir, 'model.ckpt')
        saver = tf.train.Saver()

        summary_str = None
        bno = 0
        epoch = 0
        if tf.gfile.Exists(training_dir):
            # tf.gfile.DeleteRecursively(base_dir)
            saver.restore(sess, training_dir)
            bno = sess.run(tf.train.get_or_create_global_step())
        else:
            sess.run(tf.global_variables_initializer())
            tf.gfile.MakeDirs(training_dir)

        summary, train_writer, test_writer = collect_summary(
            sess, model, summary_dir)

        while True:
            # bno = epoch*TEST_INTERVAL
            test_summary_str = None
            epoch = bno // TEST_INTERVAL
            if bno % TEST_INTERVAL == 0:
                print('{} running on test set...'.format(strftime("%H:%M:%S")))
                feeds = {data: tdata, target: tvals, seqlen: tseqlen}
                mse, worst, test_summary_str = sess.run(
                    [model.cost, model.worst, summary], feeds)
                bidx, max_diff, predict, actual = worst[0], worst[1], worst[2], worst[3]
                print('{} Epoch {} diff {:3.5f} max_diff {:3.4f} predict {} actual {} uuid {}'.format(
                    strftime("%H:%M:%S"), epoch, math.sqrt(mse), max_diff, predict, actual, tuuids[bidx]))

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
                break
            feeds = {data: trdata, target: trvals,  seqlen: trseqlen}
            summary_str, worst = sess.run(
                [summary, model.worst, model.optimize], feeds)[:-1]
            bidx, max_diff, predict, actual = worst[0], worst[1], worst[2], worst[3]
            print('{} bno {} max_diff {:3.4f} predict {} actual {}'.format(
                strftime("%H:%M:%S"), bno, max_diff, predict, actual))
            train_writer.add_summary(summary_str, bno)
            test_writer.add_summary(test_summary_str, bno)
            # train_writer.flush()
            # test_writer.flush()
            if bno % SAVE_INTERVAL == 0:
                saver.save(sess, checkpoint_file,
                           global_step=tf.train.get_global_step())
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
        saver.save(sess, checkpoint_file,
                   global_step=tf.train.get_global_step())
        # training finished, move to 'trained' folder
        trained = os.path.join(base_dir, 'trained')
        tf.gfile.MakeDirs(trained)
        tmp_dir = os.path.join(
            base_dir, '{}'.format(strftime("%Y%m%d_%H%M%S")))
        os.rename(training_dir, tmp_dir)
        shutil.move(tmp_dir, trained)
        print('{} model is saved to {}'.format(strftime("%H:%M:%S"), trained))


if __name__ == '__main__':
    run()
