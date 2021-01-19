import argparse
import os
import psutil
import math
import ray
import sys
import tensorflow as tf
import numpy as np

from time import strftime
from pathlib import Path
from corl.wc_data.input_fn import _getFtQuery, _getIndex, _init_db
from corl.wc_test.test27_mdnc import create_regressor
from corl.wcc.worker import predict_wcc

CORL_PRIOR = 100
MAX_BATCH_SIZE = 600
TIME_SHIFT = 4
MAX_STEP = 35
MIN_RCODE = 30
TOP_K = 5
COLS = ["close"]
REGRESSOR = create_regressor()


def parseArgs():
    parser = argparse.ArgumentParser()
    req = parser.add_argument_group('required named arguments')
    req.add_argument('-m', '--model', type=str,
                     help='path to the saved model', required=True)

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
    parser.add_argument('-p', '--parallel',
                        type=int,
                        help='number of parallel workers to run prediction',
                        default=math.sqrt(psutil.cpu_count(logical=True)))
    parser.add_argument('-n', '--num_cpus',
                        type=int,
                        help='number of parallel workers for data loading',
                        default=None)
    parser.add_argument('-b', '--max_batch_size',
                        type=int,
                        help='maximum number of batches in each prediction.',
                        default=MAX_BATCH_SIZE)
    parser.add_argument('-f', '--prefetch', type=int,
                        help='dataset prefetch batches', default=2)
    parser.add_argument('-i', '--init_workload', dest='init_workload', default=False,
                        action='store_true', help='initialize holistic workload for the inference')
    parser.add_argument('-g', '--gpu_grow_mem', dest='gpu_grow_mem', default=False,
                        action='store_true', help='allow gpu to allocate mem dynamically at runtime.')
    parser.add_argument('--limit_gpu_mem',
                        type=float,
                        help='pre-allocate gpu memory (in giga-bytes)',
                        default=None)
    parser.add_argument(
        '--enable_xla',
        help='enable XLA feature',
        dest='enable_xla',
        action='store_true',
    )
    parser.add_argument('--profile', dest='profile', action='store_true', default=False,
                        help='profile CG execution.')
    return parser.parse_args()


def init(args):
    ray.init(
        # num_cpus=2,  # 1.load_data; 2.save_prediction
        # webui_host='127.0.0.1',
        dashboard_host='0.0.0.0',
        log_to_driver=True,
        object_store_memory=4 * 1024 * 1024 * 1024,  # 4G
        _memory=8 * 1024 * 1024 * 1024,  # 8G
        _driver_object_store_memory=256 * 1024 * 1024,    # 256M
    )
    # _setupTensorflow(args)


def run(args):
    print("{} started inference, pid:{}".format(
        strftime("%H:%M:%S"), os.getpid()))
    _init_db(1, args.db_host, args.db_port, args.db_pwd)
    # load workload segmentation anchors from db
    # anchors = getWorkSegmentsForPrediction(
    #     CORL_PRIOR, args.db_host, args.db_port, args.db_pwd, 1)

    # in each worker, load input data from db, run model prediction, and save predictions back to wcc_predict table with bucketing
    qk, qd, qd_idx, qk2 = _getFtQuery(COLS)
    shared_args = {
        'db_host': args.db_host,
        'db_port': args.db_port,
        'db_pwd': args.db_pwd,
        'max_step': MAX_STEP,
        'time_shift': TIME_SHIFT,
        'corl_prior': CORL_PRIOR,
        'qk': qk,
        'qk2': qk2,
        'qd': qd,
        'qd_idx': qd_idx,
        'index_list': _getIndex(),
        # 'anchors': anchors,
        'args': args,
    }
    num_actors = args.num_cpus or psutil.cpu_count(logical=False)
    print("{} parallel = {}".format(
        strftime("%H:%M:%S"), args.parallel))
    print("{} num_actors = {}".format(
        strftime("%H:%M:%S"), num_actors))
    predict_wcc(num_actors,
                MIN_RCODE,
                args.max_batch_size,
                args.model,
                TOP_K,
                shared_args)


if __name__ == '__main__':
    args = parseArgs()
    init(args)
    run(args)
