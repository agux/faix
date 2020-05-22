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
from wc_data import input_file2, output_file
import math
import shutil
import random
import argparse

LOG_DIR = 'logdir'
TRACE_INTERVAL = 10
LAYER_WIDTH = 512
MEMORY_SIZE = 32
WORD_SIZE = 32
NUM_WRITES = 2
NUM_READS = 8
MAX_STEP = 35
TIME_SHIFT = 4
KEEP_PROB = 1.0

# pylint: disable-msg=E0601,E1101


def parseArgs():
    parser = argparse.ArgumentParser()
    req = parser.add_argument_group('required named arguments')
    req._parseArgs('-m', '--model', type=str,
                     help='path to the saved model', required=True)
    req._parseArgs('-r', '--rbase', type=str,
                     help='gcs remote directory path for inference file', required=True)
    req._parseArgs('-p', '--path', type=str,
                     help='gcs remote directory path for inference result files', required=True)

    parser._parseArgs('-d', '--del_used', dest='del_used', default=False,
                        action='store_true', help='delete used inference files.')
    parser._parseArgs('--project', dest='project', type=str,
                        help='gcs remote directory path for inference result files')
    parser._parseArgs('-f', '--prefetch', type=int,
                        help='dataset prefetch batches', default=2)
    parser._parseArgs('-b', '--batch', type=int, dest="batch",
                        help='inference result file batch size', default=128)
    parser._parseArgs('-g', '--gpu_grow_mem', dest='gpu_grow_mem', default=False,
                        action='store_true', help='allow gpu to allocate mem dynamically at runtime.')
    parser._parseArgs('--trace', dest='trace', action='store_true', default=False,
                        help='record full trace in validation step.')
    parser._parseArgs('--profile', dest='profile', action='store_true', default=False,
                        help='profile CG execution.')
    parser._parseArgs('--log_device', dest='log_device', action='store_true', default=False,
                        help='record device info such as CPU and GPU in tensorboard.')
    return parser.parse_args()


def run(args):
    print("{} started inference, pid:{}, del_used: {}".format(
        strftime("%H:%M:%S"), os.getpid(), args.del_used))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    keep_prob = tf.compat.v1.placeholder(tf.float32, [], name="kprob")
    config = tf.compat.v1.ConfigProto(
        log_device_placement=args.log_device,
        allow_soft_placement=True)
    config.gpu_options.allow_growth = args.gpu_grow_mem
    if not tf.io.gfile.exists(args.model):
        print('{} invalid model path: {}'.format(
            strftime("%H:%M:%S"), args.model))
        sys.exit(1)
    with tf.compat.v1.Session(config=config) as sess:
        model = dncr.DNCRegressorV1(
            layer_width=LAYER_WIDTH,
            memory_size=MEMORY_SIZE,
            word_size=WORD_SIZE,
            num_writes=NUM_WRITES,
            num_reads=NUM_READS,
            keep_prob=keep_prob
        )
        model_name = model.getName()
        print('{} using model: {}'.format(strftime("%H:%M:%S"), model_name))
        f = __file__
        file_name = f[f.rfind('/')+1:f.rindex('.py')]
        base_name = "{}_{}".format(file_name, model_name)

        ckpt = tf.train.get_checkpoint_state(args.model)
        if ckpt and ckpt.model_checkpoint_path:
            print("{} model checkpoint path: {}".format(
                strftime("%H:%M:%S"), ckpt.model_checkpoint_path))
            d = input_file2.getInferInput(args.rbase, args.prefetch)
            model.setNodes(d['features'], None, d['seqlens'], d['refs'])
            saver = tf.compat.v1.train.Saver(name="reg_saver")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("{} model checkpoint path not found, please examine the model.ckpt file under {}".format(
                strftime("%H:%M:%S"), args.model))
            sys.exit(1)

        infer_handle = sess.run(d['infer_iter'].string_handle())
        profiler = None
        if args.trace:
            print("{} full trace will be collected every {} run".format(
                strftime("%H:%M:%S"), TRACE_INTERVAL))
        if args.profile:
            profiler = tf.compat.v1.profiler.Profiler(sess.graph)
            profile_path = os.path.join(LOG_DIR, "profile")
            tf.io.gfile.makedirs(profile_path)
        bno = 0
        records = []
        indices = []
        rpaths = []
        last_rw = None
        while True:
            try:
                bno = bno + 1
                print('{} infering batch {}'.format(
                    strftime("%H:%M:%S"), bno))
                ro, rm = None, None
                if (args.trace or args.profile) and bno+1 >= 5 and bno+1 <= 10:
                    ro = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                    rm = tf.compat.v1.RunMetadata()
                code, klid, idx, rpath, r = sess.run([d['code'], d['klid'], d['idx'], d['rel_path'], model.infer],
                                                     {d['handle']: infer_handle,
                                                      keep_prob: KEEP_PROB},
                                                     options=ro, run_metadata=rm)
                posc, pcorl, negc, ncorl = r[0], r[1], r[2], r[3]
                print('{} bno {} {}@{}: posc {}, pcorl {}, negc {}, ncorl {}'.format(
                    strftime("%H:%M:%S"), bno, code, klid, posc, pcorl, negc, ncorl))
                indices.append(int(idx))
                records.append(
                    {
                        'code': str(code),
                        'klid': int(klid),
                        'positive': str(posc),
                        'pcorl': float(pcorl),
                        'negative': str(negc),
                        'ncorl': float(ncorl)
                    }
                )
                rpaths.append(str(rpath))
                if len(records) >= args.batch:
                    if last_rw is not None:
                        # wait for last write to complete
                        last_rw.result()
                    last_rw = output_file.write_result(
                        args.path, indices, records, args.del_used, rpaths)
                    indices, records, rpaths = [], [], []
                if profiler is not None and bno+1 >= 5 and bno+1 <= 10:
                    profiler.add_step(bno+1, rm)
                    if bno+1 == 10:
                        option_builder = tf.compat.v1.profiler.ProfileOptionBuilder
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
            except tf.errors.OutOfRangeError:
                print("End of Dataset.")
                break

        if len(records) > 0:
            if last_rw is not None:
                last_rw.result()
            output_file.write_result(
                args.path, indices, records, args.del_used, rpaths).result()

        output_file.shutdown()

        input_file2.check_task_status()


if __name__ == '__main__':
    args = parseArgs()
    run(args)
