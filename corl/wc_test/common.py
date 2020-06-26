# from __future__ import print_function

import os
import psutil
import shutil
import asyncio
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from time import strftime

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# from wc_data import input_fn
# from wc_data import input_bq, input_file2

import argparse

import logging
LOG_DIR = 'logdir'
LOGGER_FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(format=LOGGER_FORMAT, datefmt='[%H:%M:%S]')
log = logging.getLogger()
log.setLevel(logging.INFO)


def setupPath():
    p1 = os.path.dirname(os.path.abspath(__file__))
    p2 = os.path.dirname(p1)
    p3 = os.path.dirname(p2)
    p4 = os.path.dirname(p3)
    os.environ[
        "PYTHONPATH"] = p1 + ":" + p2 + ":" + p3 + ":" + p4 + ":" + os.environ.get(
            "PYTHONPATH", "")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir',
                        type=str,
                        help='directory path for the logdir.',
                        default=None)
    parser.add_argument('--profile_batch',
                        type=str,
                        help='batches to profile.',
                        default=None)
    parser.add_argument('--ds',
                        type=str,
                        help='datasource. such as file, db, or BigQuery.',
                        default='db')
    parser.add_argument('--dir',
                        type=str,
                        help='directory path for training and test set.')
    parser.add_argument('--parallel',
                        type=int,
                        help='database operation parallel level',
                        default=psutil.cpu_count(logical=False))
    parser.add_argument('--prefetch',
                        type=int,
                        help='dataset prefetch batches',
                        default=2)
    parser.add_argument('--db_host',
                        type=str,
                        help='database host address',
                        default=None)
    parser.add_argument('--db_port',
                        type=int,
                        help='database listening port',
                        default=None)
    parser.add_argument('--db_pwd',
                        type=str,
                        help='database password',
                        default=None)
    parser.add_argument('--vset',
                        type=int,
                        help='validation set number',
                        default=None)
    parser.add_argument('--db_pool',
                        type=int,
                        help='database connection pool size',
                        default=psutil.cpu_count(logical=False))
    parser.add_argument('--start',
                        type=int,
                        help='start training at specified batch no',
                        default=None)
    parser.add_argument('--vol_size',
                        type=int,
                        help='volume size for the dataset storage sub-folder',
                        default=None)
    parser.add_argument(
        '--terminate_on_nan',
        help='abort training process on NaN loss.',
        dest='terminate_on_nan',
        action='store_true',
    )
    parser.add_argument(
        '--enable_xla',
        help='enable XLA feature',
        dest='enable_xla',
        action='store_true',
    )
    parser.add_argument(
        '--check_input',
        help='check inputs for NaN or Inf.',
        dest='check_input',
        action='store_true',
    )
    parser.add_argument(
        '--check_weights',
        help='check trainable weights for NaN or Inf.',
        dest='check_weights',
        action='store_true',
    )
    parser.add_argument(
        '--gpu_grow_mem',
        dest='gpu_grow_mem',
        action='store_true',
        help='allow gpu to allocate mem dynamically at runtime.')
    parser.add_argument('--trace',
                        dest='trace',
                        action='store_true',
                        help='record full trace in validation step.')
    parser.add_argument('--profile',
                        dest='profile',
                        action='store_true',
                        help='profile CG execution.')
    parser.add_argument('--skip_init_test',
                        dest='skip_init_test',
                        action='store_true',
                        help='whether to skip the initial test.')
    parser.add_argument(
        '--log_device',
        dest='log_device',
        action='store_true',
        help='record device info such as CPU and GPU in tensorboard.')
    parser.add_argument('--restart',
                        help='restart training',
                        action='store_true')
    return parser.parse_args()


# def getInput(start, args):
#     ds = args.ds.lower()
#     print('{} using data source: {}'.format(strftime("%H:%M:%S"), args.ds))
#     if ds == 'db':
#         return input_fn.getInputs(TIME_SHIFT, feat_cols, MAX_STEP,
#                                   args.parallel, args.prefetch, args.db_pool,
#                                   args.db_host, args.db_port, args.db_pwd,
#                                   args.vset or VSET)
# elif ds == 'bigquery':
#     return input_bq.getInputs(start,
#                               TIME_SHIFT,
#                               feat_cols,
#                               MAX_STEP,
#                               TEST_BATCH_SIZE,
#                               vset=args.vset or VSET)
# elif ds == 'file':
#     return input_file2.getInputs(args.dir, start, args.prefetch, args.vset
#                                  or VSET, args.vol_size)
# return None


async def cleanup(dirpath, keep=5, interval=30):
    while True:
        if os.path.exists(dirpath):
            print('{} [house keeper] cleaning up {} ...'.format(
                strftime("%H:%M:%S"), dirpath))
            paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)
            # print(paths)
            dir_rm = len(paths) - keep
            if dir_rm == 0:
                print('{} [house keeper] no files to delete.'.format(
                    strftime("%H:%M:%S")))
            for i in range(dir_rm):
                print('{} [house keeper] removing {}'.format(
                    strftime("%H:%M:%S"), paths[i]))
                shutil.rmtree(paths[i], ignore_errors=True)

        await asyncio.sleep(interval)


class DebugCallback(keras.callbacks.Callback):
    def __init__(self, iterations={}, exclude_layers={}, out_file='debug.log'):
        super(DebugCallback, self).__init__()
        print('{} DebugCallback is enabled'.format(strftime("%H:%M:%S")))
        self.iterations = iterations
        self.exclude_layers = exclude_layers
        self.out_file = out_file

    def on_train_batch_end(self, batch, logs=None):
        i = self.model.optimizer.iterations.numpy()
        print('{} iteration: {}, logs={}'.format(strftime("%H:%M:%S"), i,
                                                 logs))
        if not math.isnan(logs['loss']):
            return
        print(
            '{} encountered NaN loss. checking layer weights. iteration {}, logs = {}'
            .format(strftime("%H:%M:%S"), i, logs))
        layers = self.model.layers
        for layer in layers:
            # if layer.name in {'features', 'seqlens'}:
            #     continue
            weights = layer.get_weights()
            for idx, w in enumerate(weights):
                found = False

                if np.ma.is_masked(w):
                    print(
                        'masked array found at iteration {} for {}, weight[{}]'
                        .format(i, layer, idx))
                    found = True

                nanLoc = np.argwhere(np.isnan(w))
                if len(nanLoc) > 0:
                    print(
                        'nan found at iteration {} for {}, weight[{}], location: {}'
                        .format(i, layer.name, idx, nanLoc))
                    found = True

                infLoc = np.argwhere(np.isinf(w))
                if len(infLoc) > 0:
                    print(
                        'inf found at iteration {} for {}, weight[{}], location: {}'
                        .format(i, layer.name, idx, infLoc))
                    found = True

                if found:
                    print(w)

                tf.debugging.check_numerics(
                    w, 'invalid weight found at iteration {} for {}, idx[{}]'.
                    format(i, layer.name, idx))

        # tf.print('iteration: {}'.format(i),
        #          output_stream='file://' + self.out_file)
        # tf.print(self.model.get_weights(),
        #          output_stream='file://' + self.out_file,
        #          summarize=-1)

        # if i in self.iterations:
        #     tf.print(self.model.inputs,
        #              output_stream='file://' + self.out_file,
        #              summarize=-1)
