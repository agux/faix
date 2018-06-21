from __future__ import print_function
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import argparse
import tensorflow as tf
# pylint: disable-msg=E0401
from model import base as base_model
from wc_data import input_fn
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

# pylint: disable-msg=E0601,E1101


k_cols = [
    "lr", "lr_h", "lr_o", "lr_l",
    "lr_h_c", "lr_o_c", "lr_l_c",
    "lr_ma5", "lr_ma5_h", "lr_ma5_o", "lr_ma5_l",
    "lr_ma10", "lr_ma10_h", "lr_ma10_o", "lr_ma10_l",
    "lr_vol", "lr_vol5", "lr_vol10"
]

parser = argparse.ArgumentParser()
parser.add_argument('--parallel', type=int, help='database operation parallel level',
                    default=multiprocessing.cpu_count())
parser.add_argument('--prefetch', type=int, help='dataset prefetch batches',
                    default=2)
parser.add_argument('--db_host', type=str, help='database host address',
                    default=None)
parser.add_argument('--db_port', type=int, help='database listening port',
                    default=None)
parser.add_argument(
    '--restart', help='restart training', action='store_true')
args = parser.parse_args()


def run():
    global args
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session() as sess:
        model = base_model.SRnnRegressorV3(
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
        saver = None

        summary_str = None
        d = None
        bno = 0
        epoch = 0
        restored = False
        ckpt = tf.train.get_checkpoint_state(training_dir)

        if tf.gfile.Exists(training_dir):
            print("{} training folder exists".format(strftime("%H:%M:%S")))
            if ckpt and ckpt.model_checkpoint_path:
                print("{} found model checkpoint path: {}".format(
                    strftime("%H:%M:%S"), ckpt.model_checkpoint_path))
                # Extract from checkpoint filename
                bno = int(os.path.basename(
                    ckpt.model_checkpoint_path).split('-')[1])
                print('{} resume from last training, bno = {}'.format(
                    strftime("%H:%M:%S"), bno))
                d = input_fn.getInputs(
                    bno+1, TIME_SHIFT, k_cols, MAX_STEP, args.parallel, args.prefetch)
                model.setNodes(d['uuids'], d['features'],
                               d['labels'], d['seqlens'])
                saver = tf.train.Saver()
                saver.restore(sess, training_dir)
                print('{} check restored global step: {}'.format(
                    strftime("%H:%M:%S"), sess.run(tf.train.get_global_step())))
                restored = True
            else:
                print("{} model checkpoint path not found, cleaning training folder".format(
                    strftime("%H:%M:%S")))
                tf.gfile.DeleteRecursively(training_dir)

        if not restored:
            d = input_fn.getInputs(
                bno+1, TIME_SHIFT, k_cols, MAX_STEP, args.parallel, args.prefetch)
            model.setNodes(d['uuids'], d['features'],
                           d['labels'], d['seqlens'])
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            tf.gfile.MakeDirs(training_dir)

        train_handle, test_handle = sess.run(
            [d['train_iter'].string_handle(), d['test_iter'].string_handle()])

        train_feed = {d['handle']: train_handle}
        test_feed = {d['handle']: test_handle}

        summary, train_writer, test_writer = collect_summary(
            sess, model, summary_dir)
        test_summary_str = None
        while True:
            # bno = epoch*TEST_INTERVAL
            epoch = bno // TEST_INTERVAL
            if bno % TEST_INTERVAL == 0:
                print('{} running on test set...'.format(strftime("%H:%M:%S")))
                mse, worst, test_summary_str = sess.run(
                    [model.cost, model.worst, summary], test_feed)
                uuid, max_diff, predict, actual = worst[0], worst[1], worst[2], worst[3]
                print('{} Epoch {} diff {:3.5f} max_diff {:3.4f} predict {} actual {} uuid {}'.format(
                    strftime("%H:%M:%S"), epoch, math.sqrt(mse), max_diff, predict, actual, uuid))
            try:
                print('{} training batch {}'.format(
                    strftime("%H:%M:%S"), bno+1))
                summary_str, worst = sess.run(
                    [summary, model.worst, model.optimize], train_feed)[:-1]
            except tf.errors.OutOfRangeError:
                print("End of Dataset.")
                break
            bno = bno+1
            _, max_diff, predict, actual = worst[0], worst[1], worst[2], worst[3]
            print('{} bno {} max_diff {:3.4f} predict {} actual {}'.format(
                strftime("%H:%M:%S"), bno, max_diff, predict, actual))
            train_writer.add_summary(summary_str, bno)
            test_writer.add_summary(test_summary_str, bno)
            train_writer.flush()
            test_writer.flush()
            if bno == 1 or bno % SAVE_INTERVAL == 0:
                saver.save(sess, checkpoint_file,
                           global_step=tf.train.get_global_step())
        # test last epoch
        print('{} running on test set...'.format(strftime("%H:%M:%S")))
        mse, worst, test_summary_str = sess.run(
            [model.cost, model.worst, summary], test_feed)
        uuid, max_diff, predict, actual = worst[0], worst[1], worst[2], worst[3]
        print('{} Epoch {} diff {:3.5f} max_diff {:3.4f} predict {} actual {} uuid {}'.format(
            strftime("%H:%M:%S"), epoch, math.sqrt(mse), max_diff, predict, actual, uuid))
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
