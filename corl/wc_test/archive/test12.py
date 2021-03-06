from __future__ import print_function
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import tensorflow as tf
# pylint: disable-msg=E0401
from model import drnn_regressor as drnn
from wc_data import input_fn, input_bq, input_file
from time import strftime
from test11 import parseArgs, feat_cols, LOG_DIR, collect_summary, getInput
import argparse
import numpy as np
import math
import multiprocessing
import shutil
import random

VSET = 9
TEST_BATCH_SIZE = 3000
TEST_INTERVAL = 50
SAVE_INTERVAL = 10
LAYER_WIDTH = 512
MAX_STEP = 35
TIME_SHIFT = 4
DIM = 3
KEEP_PROB = 0.5
LEARNING_RATE = 1e-3
DECAYED_LEARNING_RATE = 1e-4
DECAYED_LR_START = 0.7
SEED = 285139

# pylint: disable-msg=E0601,E1101

bst_saver, bst_score, bst_file, bst_ckpt = None, None, None, None


def validate(sess, model, summary, feed, bno, epoch):
    global bst_saver, bst_score, bst_file, bst_ckpt
    print('{} running on test set...'.format(strftime("%H:%M:%S")))
    mse, worst, test_summary_str = sess.run(
        [model.cost, model.worst, summary], feed)
    diff, max_diff, predict, actual = math.sqrt(
        mse), worst[0], worst[1], worst[2]
    print('{} Epoch {} diff {:3.5f} max_diff {:3.4f} predict {} actual {}'.format(
        strftime("%H:%M:%S"), epoch, diff, max_diff, predict, actual))
    found_better = False
    if diff < bst_score:
        bst_score = diff
        bst_file.seek(0)
        bst_file.truncate()
        bst_file.write('{}\n{}\n'.format(diff, bno))
        bst_file.truncate()
        bst_saver.save(sess, bst_ckpt,
                       global_step=tf.compat.v1.train.get_global_step())
        print('{} acquired better model with validation score {}, at batch {}'.format(
              strftime("%H:%M:%S"), diff, bno))
        found_better = True
    return test_summary_str, found_better


def run(args):
    global bst_saver, bst_score, bst_file, bst_ckpt
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    random.seed(SEED)
    keep_prob = tf.compat.v1.placeholder(tf.float32, [], name="keep_prob")
    learning_rate = tf.compat.v1.placeholder(tf.float32, [], name="learning_rate")
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
        model = drnn.DRnnRegressorV6(
            dim=DIM,
            keep_prob=keep_prob,
            layer_width=LAYER_WIDTH,
            learning_rate=learning_rate)
        model_name = model.getName()
        print('{} using model: {}'.format(strftime("%H:%M:%S"), model_name))
        f = __file__
        testn = f[f.rfind('/')+1:f.rindex('.py')]
        base_dir = '{}/{}_{}'.format(LOG_DIR, testn, model_name)
        training_dir = os.path.join(base_dir, 'training')
        checkpoint_file = os.path.join(training_dir, 'model.ckpt')
        bst_ckpt = os.path.join(base_dir, 'best', 'model.ckpt')
        bst_file_path = os.path.join(base_dir, 'best_score')
        saver = None
        summary_str = None
        d = None
        restored = False
        bno, epoch, bst_score = 0, 0, sys.maxint
        ckpt = tf.train.get_checkpoint_state(training_dir)

        if tf.io.gfile.exists(bst_file_path):
            bst_file = open(bst_file_path, 'r+')
            bst_file.seek(0)
            try:
                bst_score = float(bst_file.readline().rstrip())
                print('{} previous best score: {}'.format(
                    strftime("%H:%M:%S"), bst_score))
            except Exception:
                print('{} not able to read best score. best_score file is invalid.'.format(
                    strftime("%H:%M:%S")))
            bst_file.seek(0)

        if tf.io.gfile.exists(training_dir):
            print("{} training folder exists".format(strftime("%H:%M:%S")))
            if ckpt and ckpt.model_checkpoint_path:
                print("{} found model checkpoint path: {}".format(
                    strftime("%H:%M:%S"), ckpt.model_checkpoint_path))
                # Extract from checkpoint filename
                bno = int(os.path.basename(
                    ckpt.model_checkpoint_path).split('-')[1])
                print('{} resuming from last training, bno = {}'.format(
                    strftime("%H:%M:%S"), bno))
                d = getInput(bno+1, args)
                model.setNodes(d['features'], d['labels'], d['seqlens'])
                saver = tf.compat.v1.train.Saver(name="reg_saver")
                saver.restore(sess, ckpt.model_checkpoint_path)
                restored = True
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
            d = getInput(bno+1, args)
            model.setNodes(d['features'], d['labels'], d['seqlens'])
            saver = tf.compat.v1.train.Saver(name="reg_saver")
            sess.run(tf.compat.v1.global_variables_initializer())
            tf.io.gfile.makedirs(training_dir)
            bst_file = open(bst_file_path, 'w+')
        bst_saver = tf.compat.v1.train.Saver(name="bst_saver")

        train_handle, test_handle = sess.run(
            [d['train_iter'].string_handle(), d['test_iter'].string_handle()])

        summary, train_writer, test_writer = collect_summary(
            sess, model, training_dir)
        test_summary_str = None
        lr = LEARNING_RATE
        while True:
            # bno = epoch*TEST_INTERVAL
            epoch = bno // TEST_INTERVAL
            found_better = False
            if restored or bno % TEST_INTERVAL == 0:
                test_summary_str, found_better = validate(
                    sess, model, summary, {d['handle']: test_handle, keep_prob: 1, learning_rate: LEARNING_RATE}, bno, epoch)
                restored = False
            try:
                kp = min(1, random.uniform(KEEP_PROB, 1.05))
                if bno > int(DECAYED_LR_START * bno) and lr != DECAYED_LEARNING_RATE and found_better:
                    lr = DECAYED_LEARNING_RATE
                print('{} training batch {}, random keep_prob:{}, learning_rate:{}'.format(
                    strftime("%H:%M:%S"), bno+1, kp, lr))
                summary_str, worst = sess.run(
                    [summary, model.worst, model.optimize], {d['handle']: train_handle, keep_prob: kp, learning_rate: lr})[:-1]
            except tf.errors.OutOfRangeError:
                print("End of Dataset.")
                break
            bno = bno+1
            max_diff, predict, actual = worst[0], worst[1], worst[2]
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
        test_summary_str, _ = validate(
            sess, model, summary, {d['handle']: test_handle, keep_prob: 1, learning_rate: LEARNING_RATE}, bno, epoch)
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
