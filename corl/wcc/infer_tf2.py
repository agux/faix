import argparse
import os
import psutil
import math
import ray
import tensorflow as tf
import numpy as np

from time import strftime
from pathlib import Path
from corl.wc_data.input_fn import getWorkloadForPrediction, _getFtQuery, _getIndex
from corl.wc_test.test27_mdnc import create_regressor
from corl.wcc.worker import predict_wcc

CORL_PRIOR = 100
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
    parser.add_argument('-f', '--prefetch', type=int,
                        help='dataset prefetch batches', default=2)
    parser.add_argument('-g', '--gpu_grow_mem', dest='gpu_grow_mem', default=False,
                        action='store_true', help='allow gpu to allocate mem dynamically at runtime.')
    parser.add_argument('--profile', dest='profile', action='store_true', default=False,
                        help='profile CG execution.')
    return parser.parse_args()



def run(args):
    print("{} started inference, pid:{}".format(
        strftime("%H:%M:%S"), os.getpid()))
    # load total workload from db
    workload = getWorkloadForPrediction(
        CORL_PRIOR, args.db_host, args.db_port, args.db_pwd)
    # delegate to ray remote workers with split-even workloads
    work_seg = np.array_split(workload, args.parallel)
    # in each worker, load input data from db, run model prediction, and save predictions back to wcc_predict table with bucketing
    qk, qd, qd_idx = _getFtQuery(COLS)
    shared_args = ray.put({
        'db_host': args.db_host,
        'db_port': args.db_port,
        'db_pwd': args.db_pwd,
        'max_step': MAX_STEP,
        'time_shift': TIME_SHIFT,
        'qk': qk,
        'qd': qd,
        'qd_idx': qd_idx,
        'index_list': _getIndex(),
    })
    shared_args_oid = ray.put([shared_args])
    tasks = [predict_wcc.remote(
        work, MIN_RCODE, args.model, TOP_K, shared_args, shared_args_oid) for work in work_seg]
    ray.get(tasks)


if __name__ == '__main__':
    args = parseArgs()
    run(args)
