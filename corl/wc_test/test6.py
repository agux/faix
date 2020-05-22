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
from time import strftime
from test5 import collect_summary
import os
import numpy as np
import math
import multiprocessing
import shutil

# N_TEST = 100
VSET = 7
TEST_INTERVAL = 50
SAVE_INTERVAL = 10
LAYER_WIDTH = 512
MAX_STEP = 35
TIME_SHIFT = 4
DIM = 5
DROP_OUT = 0.5
LEARNING_RATE = 1e-3
LOG_DIR = 'logdir'

# pylint: disable-msg=E0601,E1101

def parseArgs():
    parser = argparse.ArgumentParser()
    parser._parseArgs('--ds', type=str, help='datasource. such as db, or BigQuery.',
                        default='db')
    parser._parseArgs('--parallel', type=int, help='database operation parallel level',
                        default=multiprocessing.cpu_count())
    parser._parseArgs('--prefetch', type=int, help='dataset prefetch batches',
                        default=2)
    parser._parseArgs('--db_host', type=str, help='database host address',
                        default=None)
    parser._parseArgs('--db_port', type=int, help='database listening port',
                        default=None)
    parser._parseArgs('--db_pwd', type=str, help='database password',
                        default=None)
    parser._parseArgs('--vset', type=int, help='validation set number',
                        default=None)
    parser._parseArgs('--db_pool', type=int, help='database connection pool size',
                        default=multiprocessing.cpu_count())
    parser._parseArgs('--start', type=int, help='start training at specified batch no',
                        default=None)
    parser._parseArgs(
        '--restart', help='restart training', action='store_true')
    return parser.parse_args()


bst_saver, bst_score, bst_file, bst_ckpt = None, None, None, None

def validate(sess, model, summary, feed, bno, epoch):
    global bst_saver, bst_score, bst_file, bst_ckpt
    print('{} running on test set...'.format(strftime("%H:%M:%S")))
    mse, worst, test_summary_str = sess.run(
        [model.cost, model.worst, summary], feed)
    diff, uuid, max_diff, predict, actual = math.sqrt(
        mse), worst[0], worst[1], worst[2], worst[3]
    print('{} Epoch {} diff {:3.5f} max_diff {:3.4f} predict {} actual {} uuid {}'.format(
        strftime("%H:%M:%S"), epoch, diff, max_diff, predict, actual, uuid))
    if diff < bst_score:
        bst_score = diff
        bst_file.seek(0)
        bst_file.write('{}\n{}\n'.format(diff, bno))
        bst_file.truncate()
        bst_saver.save(sess, bst_ckpt,
                       global_step=tf.compat.v1.train.get_global_step())
        print('{} acquired better model with validation score {}, at batch {}'.format(
              strftime("%H:%M:%S"), diff, bno))
    return test_summary_str


def run(args):
    global bst_saver, bst_score, bst_file, bst_ckpt
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    dropout = tf.compat.v1.placeholder(tf.float32, [], name="dropout")
    with tf.compat.v1.Session() as sess:
        model = base_model.SRnnRegressorV4(
            dim=DIM,
            dropout=dropout,
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
        bst_ckpt = os.path.join(training_dir, 'best', 'model.ckpt')
        saver = None
        summary_str = None
        d = None
        restored = False
        bno, epoch, bst_score = 0, 0, sys.maxint
        ckpt = tf.train.get_checkpoint_state(training_dir)

        if tf.io.gfile.exists(training_dir):
            print("{} training folder exists".format(strftime("%H:%M:%S")))
            bst_file = open(os.path.join(training_dir, 'best_score'), 'w+')
            bst_file.seek(0)
            if ckpt and ckpt.model_checkpoint_path:
                print("{} found model checkpoint path: {}".format(
                    strftime("%H:%M:%S"), ckpt.model_checkpoint_path))
                # Extract from checkpoint filename
                bno = int(os.path.basename(
                    ckpt.model_checkpoint_path).split('-')[1])
                print('{} resuming from last training, bno = {}'.format(
                    strftime("%H:%M:%S"), bno))
                d = input_fn.getInputs(
                    bno+1, TIME_SHIFT, None, MAX_STEP, args.parallel,
                    args.prefetch, args.db_pool, args.db_host, args.db_port, args.db_pwd, args.vset or VSET)
                model.setNodes(d['uuids'], d['features'],
                               d['labels'], d['seqlens'])
                saver = tf.compat.v1.train.Saver(name="reg_saver")
                saver.restore(sess, ckpt.model_checkpoint_path)
                restored = True
                bst_score = bst_file.readline().rstrip()
                print('{} previous best score: {}'.format(
                    strftime("%H:%M:%S"), bst_score))
                rbno = sess.run(tf.compat.v1.train.get_global_step())
                print('{} check restored global step: {}, previous batch no: {}'.format(
                    strftime("%H:%M:%S"), rbno, bno))
                if bno != rbno:
                    print('{} bno({}) inconsistent with global step({}). reset global step with bno.'.format(
                        strftime("%H:%M:%S"), bno, rbno))
                    gstep = tf.compat.v1.train.get_global_step(sess.graph)
                    sess.run(tf.compat.v1.assign(gstep, bno))
            else:
                print("{} model checkpoint path not found, cleaning training folder".format(
                    strftime("%H:%M:%S")))
                tf.io.gfile.rmtree(training_dir)

        if not restored:
            d = input_fn.getInputs(
                bno+1, TIME_SHIFT, None, MAX_STEP, args.parallel,
                args.prefetch, args.db_pool, args.db_host, args.db_port, args.db_pwd, args.vset or VSET)
            model.setNodes(d['uuids'], d['features'],
                           d['labels'], d['seqlens'])
            saver = tf.compat.v1.train.Saver(name="reg_saver")
            sess.run(tf.compat.v1.global_variables_initializer())
            tf.io.gfile.makedirs(training_dir)
            bst_file = open(os.path.join(training_dir, 'best_score'), 'w+')
        bst_saver = tf.compat.v1.train.Saver(name="bst_saver")

        train_handle, test_handle = sess.run(
            [d['train_iter'].string_handle(), d['test_iter'].string_handle()])

        train_feed = {d['handle']: train_handle, dropout: DROP_OUT}
        test_feed = {d['handle']: test_handle, dropout: 0}

        summary, train_writer, test_writer = collect_summary(
            sess, model, summary_dir)
        test_summary_str = None
        while True:
            # bno = epoch*TEST_INTERVAL
            epoch = bno // TEST_INTERVAL
            if restored or bno % TEST_INTERVAL == 0:
                test_summary_str = validate(
                    sess, model, summary, test_feed, bno, epoch)
                restored = False
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
                           global_step=tf.compat.v1.train.get_global_step())
        # test last epoch
        test_summary_str = validate(
            sess, model, summary, test_feed, bno, epoch)
        train_writer.add_summary(summary_str, bno)
        test_writer.add_summary(test_summary_str, bno)
        train_writer.flush()
        test_writer.flush()
        saver.save(sess, checkpoint_file,
                   global_step=tf.compat.v1.train.get_global_step())
        # training finished, move to 'trained' folder
        trained = os.path.join(base_dir, 'trained')
        tf.io.gfile.makedirs(trained)
        tmp_dir = os.path.join(
            base_dir, strftime("%Y%m%d_%H%M%S"))
        os.rename(training_dir, tmp_dir)
        shutil.move(tmp_dir, trained)
        print('{} model is saved to {}'.format(strftime("%H:%M:%S"), trained))
        bst_file.close()


if __name__ == '__main__':
    args = parseArgs()
    run(args)
