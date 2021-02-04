import os
import psutil
import math
import numpy as np
import tensorflow as tf
import tracemalloc
import argparse
import logging
from tensorflow import keras
from pathlib import Path
from time import strftime

LOG_DIR = 'logdir'
LOGGER_FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(format=LOGGER_FORMAT, datefmt='[%H:%M:%S]')
log = logging.getLogger()
log.setLevel(logging.INFO)


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
    parser.add_argument('--tracemalloc',
                        type=str,
                        help='specify the batches for snapshots',
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
    parser.add_argument('--limit_gpu_mem',
                        type=float,
                        help='pre-allocate gpu memory (in giga-bytes)',
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


def setupPath():
    p1 = os.path.dirname(os.path.abspath(__file__))
    p2 = os.path.dirname(p1)
    p3 = os.path.dirname(p2)
    p4 = os.path.dirname(p3)
    os.environ[
        "PYTHONPATH"] = p1 + ":" + p2 + ":" + p3 + ":" + p4 + ":" + os.environ.get(
            "PYTHONPATH", "")


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

class TracemallocCallback(keras.callbacks.Callback):
    def __init__(self, nframe=500, batches='200,300', out_file='tracemalloc.log'):
        super(TracemallocCallback, self).__init__()
        tracemalloc.start(nframe)
        print('{} TracemallocCallback is enabled at batches {}'.format(
            strftime("%H:%M:%S"), batches))
        seg = batches.split(',')
        self.start = int(seg[0])
        self.end = int(seg[1])
        self.out_file = out_file
        path_seg = os.path.splitext(self.out_file)
        self.out_file_base, self.out_file_ext = path_seg[0], path_seg[1]

    def on_train_batch_end(self, batch, logs=None):
        i = self.model.optimizer.iterations.numpy()
        if i == self.start:
            self.snapshot1 = tracemalloc.take_snapshot()
            dest = '{}_1{}'.format(self.out_file_base, self.out_file_ext)
            self.snapshot1.dump(dest)
            tf.print('tracemalloc snapshot #1 at iteration ',
                     i, ' has been dumped to ', dest)
        elif i == self.end:
            dest = '{}_2{}'.format(self.out_file_base, self.out_file_ext)
            snapshot2 = tracemalloc.take_snapshot()
            snapshot2.dump(dest)
            tf.print('tracemalloc snapshot #2 at iteration ',
                     i, ' has been dumped to ', dest)
            stats_diff = snapshot2.compare_to(self.snapshot1, 'lineno')
            diff_dest = '{}_d{}'.format(self.out_file_base, self.out_file_ext)
            with open(diff_dest, 'w') as f:
                for stat in stats_diff:
                    print(stat, file=f)
                tf.print('2 snapshot compare has been dumped to ', diff_dest)
