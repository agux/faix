# High capacity DNC regressor with gradient-checkpointing

from __future__ import print_function
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import tensorflow as tf
# pylint: disable-msg=E0401
from model import dnc_regressor as dncr
from time import strftime
from test11 import parseArgs, feat_cols, LOG_DIR, collect_summary
from test13 import getInput
import math
import shutil
import random

VSET = 9
TEST_BATCH_SIZE = 3000
TEST_INTERVAL = 50
TRACE_INTERVAL = 10
SAVE_INTERVAL = 20
PARALLEL_ITERATIONS = 6
LAYER_WIDTH = 512
MEMORY_SIZE = 64
WORD_SIZE = 64
NUM_WRITES = 4
NUM_READS = 16
CLIP_VALUE = 50
MAX_GRAD_NORM = 100
MAX_STEP = 35
TIME_SHIFT = 4
KEEP_PROB = 0.5
LEARNING_RATE = 1e-3
LR_DECAY_STEPS = 1000
DECAYED_LR_START = 40000
DROPOUT_DECAY_STEPS = 1000
DECAYED_DROPOUT_START = 40000
SEED = 285139

# pylint: disable-msg=E0601,E1101

bst_saver, bst_score, bst_file, bst_ckpt = None, None, None, None


def validate(sess, model, summary, feed, bno, epoch):
    global bst_saver, bst_score, bst_file, bst_ckpt
    print('{} running on test set...'.format(strftime("%H:%M:%S")))
    mse, worst, test_summary_str = sess.run(
        [model.cost, model.worst, summary], feed
    )
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
                       global_step=tf.train.get_global_step())
        print('{} acquired better model with validation score {}, at batch {}'.format(
              strftime("%H:%M:%S"), diff, bno))
        found_better = True
    return test_summary_str, found_better


def run(args):
    global bst_saver, bst_score, bst_file, bst_ckpt
    print("{} started training, pid:{}".format(
        strftime("%H:%M:%S"), os.getpid()))
    tf.logging.set_verbosity(tf.logging.INFO)
    keep_prob = tf.placeholder(tf.float32, [], name="kprob")
    config = tf.ConfigProto(
        log_device_placement=args.log_device,
        allow_soft_placement=True)
    config.gpu_options.allow_growth = args.gpu_grow_mem
    with tf.Session(config=config) as sess:
        model = dncr.DNCRegressorV2(
            layer_width=LAYER_WIDTH,
            memory_size=MEMORY_SIZE,
            word_size=WORD_SIZE,
            num_writes=NUM_WRITES,
            num_reads=NUM_READS,
            clip_value=CLIP_VALUE,
            max_grad_norm=MAX_GRAD_NORM,
            keep_prob=keep_prob,
            decayed_dropout_start=DECAYED_DROPOUT_START,
            dropout_decay_steps=DROPOUT_DECAY_STEPS,
            learning_rate=LEARNING_RATE,
            decayed_lr_start=DECAYED_LR_START,
            lr_decay_steps=LR_DECAY_STEPS,
            seed=SEED,
            parallel_iterations=PARALLEL_ITERATIONS)
        model_name = model.getName()
        print('{} using model: {}'.format(strftime("%H:%M:%S"), model_name))
        f = __file__
        testn = f[f.rfind('/')+1:f.rindex('.py')]
        base_name = "{}_{}".format(testn, model_name)
        base_dir = '{}/{}'.format(LOG_DIR, base_name)
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

        if tf.gfile.Exists(bst_file_path):
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

        if tf.gfile.Exists(training_dir):
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
                saver = tf.train.Saver(name="reg_saver")
                saver.restore(sess, ckpt.model_checkpoint_path)
                restored = True
                rbno = sess.run(tf.train.get_global_step())
                print('{} check restored global step: {}, previous batch no: {}'.format(
                    strftime("%H:%M:%S"), rbno, bno))
                if bno != rbno:
                    print('{} bno({}) inconsistent with global step({}). reset global step with bno.'.format(
                        strftime("%H:%M:%S"), bno, rbno))
                    gstep = tf.train.get_global_step(sess.graph)
                    sess.run(tf.assign(gstep, bno))
            else:
                print("{} model checkpoint path not found, cleaning training folder".format(
                    strftime("%H:%M:%S")))
                tf.gfile.DeleteRecursively(training_dir)

        if not restored:
            d = getInput(bno+1, args)
            model.setNodes(d['features'], d['labels'], d['seqlens'])
            saver = tf.train.Saver(name="reg_saver")
            sess.run(tf.global_variables_initializer())
            tf.gfile.MakeDirs(training_dir)
            bst_file = open(bst_file_path, 'w+')
        bst_saver = tf.train.Saver(name="bst_saver")

        train_handle, test_handle = sess.run(
            [d['train_iter'].string_handle(), d['test_iter'].string_handle()])

        summary, train_writer, test_writer = collect_summary(
            sess, model, training_dir)
        profiler = None
        profile_path = None
        test_summary_str = None
        if args.trace:
            print("{} full trace will be collected every {} run".format(
                strftime("%H:%M:%S"), TRACE_INTERVAL))
        if args.profile:
            profiler = tf.profiler.Profiler(sess.graph)
            profile_path = os.path.join(LOG_DIR, "profile")
            tf.gfile.MakeDirs(profile_path)
        while True:
            epoch = bno // TEST_INTERVAL
            if (restored or bno % TEST_INTERVAL == 0) and not (args.skip_init_test and bno == 0):
                test_summary_str, _ = validate(
                    sess, model, summary, {
                        d['handle']: test_handle, keep_prob: 1.0},
                    bno, epoch)
                restored = False
            lr, kp = None, None
            try:
                print('{} training batch {}'.format(
                    strftime("%H:%M:%S"), bno+1))
                ro, rm = None, None
                if (args.trace or args.profile) and bno+1 >= 5 and bno+1 <= 10:
                    ro = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    rm = tf.RunMetadata()
                summary_str, kp, lr, worst = sess.run(
                    [summary, model.keep_prob, model.learning_rate,
                        model.worst, model.optimize],
                    {d['handle']: train_handle, keep_prob: KEEP_PROB},
                    options=ro, run_metadata=rm)[:-1]
                if profiler is not None and bno+1 >= 5 and bno+1 <= 10:
                    profiler.add_step(bno+1, rm)
                    if bno+1 == 10:
                        option_builder = tf.profiler.ProfileOptionBuilder
                        # profile timing of model operations
                        opts = (option_builder(option_builder.time_and_memory())
                                .with_step(-1)
                                .with_file_output(os.path.join(profile_path, "{}_ops.txt".format(base_name)))
                                .select(['micros', 'bytes', 'occurrence'])
                                .order_by('micros')
                                .build())
                        profiler.profile_operations(options=opts)
                        # profile timing by model name scope
                        opts = (option_builder(option_builder.time_and_memory())
                                .with_step(-1)
                                .with_file_output(os.path.join(profile_path, "{}_scope.txt".format(base_name)))
                                .select(['micros', 'bytes', 'occurrence'])
                                .order_by('micros')
                                .build())
                        profiler.profile_name_scope(options=opts)
                        # generate timeline graph
                        opts = (option_builder(option_builder.time_and_memory())
                                .with_step(bno+1)
                                .with_timeline_output(os.path.join(profile_path, "{}_timeline.json".format(base_name)))
                                .build())
                        profiler.profile_graph(options=opts)
                        # Auto detect problems and generate advice.
                        # opts = (option_builder(option_builder.time_and_memory()).
                        #         with_file_output(os.path.join(profile_path, "{}_advise.txt".format(base_name))).
                        #         build())
                        # profiler.advise(options=opts)
            except tf.errors.OutOfRangeError:
                print("End of Dataset.")
                break
            bno = bno+1
            max_diff, predict, actual = worst[0], worst[1], worst[2]
            print('{} bno {} lr: {:1.6f}, kp: {:1.5f}, max_diff {:3.4f} predict {} actual {}'.format(
                strftime("%H:%M:%S"), bno, lr, kp, max_diff, predict, actual))
            train_writer.add_summary(summary_str, bno)
            if rm is not None:
                train_writer.add_run_metadata(
                    rm, "bno_{}".format(bno))
            train_writer.flush()
            if test_summary_str is not None:
                test_writer.add_summary(test_summary_str, bno)
                test_writer.flush()
            if bno == 1 or bno % SAVE_INTERVAL == 0:
                saver.save(sess, checkpoint_file,
                           global_step=tf.train.get_global_step())
        # test last epoch
        test_summary_str, _ = validate(
            sess, model, summary,
            {d['handle']: test_handle, keep_prob: 1.0},
            bno, epoch)
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
            base_dir, strftime("%Y%m%d_%H%M%S"))
        os.rename(training_dir, tmp_dir)
        shutil.move(tmp_dir, trained)
        print('{} model is saved to {}'.format(strftime("%H:%M:%S"), trained))
        bst_file.close()


if __name__ == '__main__':
    args = parseArgs()
    run(args)
