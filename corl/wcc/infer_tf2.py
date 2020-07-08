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
from .worker import predict_wcc

CORL_PRIOR = 100
TIME_SHIFT = 4
MAX_STEP = 35
MIN_RCODE = 30
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


def _load_model(model_path):
    if not os.path.exists(model_path):
        raise Exception(
            "{} is not a valid path for the model".format(model_path))

    ckpts = sorted(Path(model_path).iterdir(), key=os.path.getmtime)
    model = None
    if len(ckpts) > 0:
        print("{} model folder exists. #checkpoints: {}".format(
            strftime("%H:%M:%S"), len(ckpts)))
        ck_path = tf.train.latest_checkpoint(model_path)
        print("{} latest model checkpoint path: {}".format(
            strftime("%H:%M:%S"), ck_path))
        # Extract from checkpoint filename
        model = REGRESSOR.getModel()
        model.load_weights(str(ck_path))
        REGRESSOR.compile()
    return model


def run(args):
    print("{} started inference, pid:{}".format(
        strftime("%H:%M:%S"), os.getpid()))
    # TODO re-write inference process
    # load total workload from db
    workload = getWorkloadForPrediction(
        CORL_PRIOR, args.db_host, args.db_port, args.db_pwd)
    # delegate to ray remote workers with split-even workloads
    work_seg = np.array_split(workload, args.parallel)
    # in each worker, load input data from db, run model prediction, and save predictions back to stockrel table with bucketing
    qk, qd, qd_idx = _getFtQuery()
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
    model = _load_model(args.model)
    tasks = [predict_wcc.remote(
        work, MIN_RCODE, model, shared_args) for work in work_seg]
    ray.get(tasks)


if __name__ == '__main__':
    args = parseArgs()
    run(args)
