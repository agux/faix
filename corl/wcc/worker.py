import ray
import sys
import os
import time

import numpy as np
import tensorflow as tf

from ray.util.queue import Queue
from time import strftime, sleep
from pathlib import Path
from mysql.connector.pooling import MySQLConnectionPool
from corl.wc_data.series import DataLoader, getSeries_v2
from corl.wc_test.test27_mdnc import create_regressor
from corl.wc_data.input_fn import getWorkloadForPredictionFromTags

REGRESSOR = create_regressor()
cnxpool = None
BUCKET_SIZE = 64
bucket = []
MAX_K = 5
WCC_INSERT = """
    INSERT INTO `secu`.`wcc_predict`
        (`code`,`date`,`klid`,
        `t1_code`,`t2_code`,`t3_code`,`t4_code`,`t5_code`,
        `t1_corl`,`t2_corl`,`t3_corl`,`t4_corl`,`t5_corl`,
        `b1_code`,`b2_code`,`b3_code`,`b4_code`,`b5_code`,
        `b1_corl`,`b2_corl`,`b3_corl`,`b4_corl`,`b5_corl`,
        `rcode_size`,`udate`,`utime`)
    VALUES
        (%s,%s,%s,
        %s,%s,%s,%s,%s,
        %s,%s,%s,%s,%s,
        %s,%s,%s,%s,%s,
        %s,%s,%s,%s,%s,
        %s,%s,%s)
"""


def _init(db_pool_size=None, db_host=None, db_port=None, db_pwd=None):
    # FIXME too many db initialization message in the log and 'aborted clients' in mysql dashboard
    global cnxpool
    size = db_pool_size or 5
    print("{} initializing mysql connection pool of size {}".format(
        strftime("%H:%M:%S"), size))
    cnxpool = MySQLConnectionPool(
        pool_name="dbpool",
        pool_size=size,
        host=db_host or '127.0.0.1',
        port=db_port or 3306,
        user='mysql',
        database='secu',
        password=db_pwd or '123456',
        # ssl_ca='',
        # use_pure=True,
        connect_timeout=90000)


def _get_rcodes_for(code, table, dates):
    # search for reference codes by matching dates
    ym = {d.replace('-', '')[:-2]
          for d in dates}  # extract year-month to a set
    query = ("select code from {} "
             "where ym in ({}) "
             "and code <> %s "
             "and date in ({}) "
             "group by code "
             "having count(*) = %s"
             ).format(
        table,
        ",".join(ym),
        ','.join(['%s']*len(dates))
    )
    cnx = cnxpool.get_connection()
    cursor = None
    try:
        cursor = cnx.cursor(buffered=True)
        cursor.execute(
            query,
            (code, *dates, len(dates))
        )
        count = cursor.rowcount
        if count == 0:
            return []
        rows = cursor.fetchall()
        return [r[0] for r in rows]
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        if cursor is not None:
            cursor.close()
        cnx.close()


def _get_rcodes(code, klid, steps, shift):
    global cnxpool
    start = klid - steps - shift + 1
    cnx = cnxpool.get_connection()
    cursor = None
    try:
        cursor = cnx.cursor(buffered=True)
        cursor.execute(
            'SELECT date FROM kline_d_b WHERE code = %s and klid between %s and %s ORDER BY klid',
            (code, start, klid))
        count = cursor.rowcount
        if count == 0:
            print(
                '{} {}@({},{}) no data in kline_d_b'.format(
                    strftime("%H:%M:%S"),
                    code, start, klid
                )
            )
            return None
        elif count < steps+shift:
            print(
                '{} [severe] {}@({},{}) some kline data may be missing. skipping'.format(
                    strftime("%H:%M:%S"),
                    code, start, klid
                )
            )
            return None
        rows = cursor.fetchall()
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        if cursor is not None:
            cursor.close()
        cnx.close()

    dates = [r[0] for r in rows]
    # get rcodes from kline table
    rcodes_k = _get_rcodes_for(code, 'kline_d_b_lr', dates)
    # get rcodes from index table
    rcodes_i = _get_rcodes_for(code, 'index_d_n_lr', dates)
    return rcodes_k + rcodes_i


def _process(pool, code, klid, date_start, date_end, min_rcode, shared_args):
    # find eligible rcodes
    rcodes = _get_rcodes(
        code, klid, shared_args['max_step'], shared_args['time_shift'])
    # check rcodes
    if len(rcodes) < min_rcode:
        print('{} {}@({},{},{}) has {} eligible reference codes, skipping'.format(
            strftime("%H:%M:%S"),
            code, klid, date_start, date_end, len(rcodes),
        ))
        return [], []
    tasks = pool.map(
        lambda a, rcode: a.get_series.remote(
            code, klid, date_start, date_end, rcode, None),
        rcodes
    )
    return np.array(list(tasks), np.float32), np.array(rcodes, object)


def _save_prediction(code=None, klid=None, date=None, rcodes=None, top_k=None, predictions=None, udate=None, utime=None):
    global bucket, cnxpool
    if code is not None:
        # get top and bottom k
        top_k = top_k if top_k <= MAX_K else MAX_K
        p = predictions
        top_idx = np.argpartition(p, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(p[top_idx])]
        top_k_corl = p[top_idx][::-1]
        top_k_code = rcodes[top_idx][::-1]
        bottom_idx = np.argpartition(p, top_k)[: top_k]
        bottom_idx = bottom_idx[np.argsort(p[bottom_idx])]
        bottom_k_corl = p[bottom_idx]
        bottom_k_code = rcodes[bottom_idx]
        if top_k < MAX_K:
            pad = MAX_K-top_k
            top_k_corl = np.pad(top_k_corl, ((0, 0), (0, pad)),
                                mode='constant', constant_values=None)
            top_k_code = np.pad(top_k_code, ((0, 0), (0, pad)),
                                mode='constant', constant_values=None)
            bottom_k_corl = np.pad(bottom_k_corl, ((0, 0), (0, pad)),
                                   mode='constant', constant_values=None)
            bottom_k_code = np.pad(bottom_k_code, ((0, 0), (0, pad)),
                                   mode='constant', constant_values=None)
        bucket.append((
            code, date, klid,
            *top_k_code,
            *(top_k_corl.tolist()),
            *bottom_k_code,
            *(bottom_k_corl.tolist()),
            len(rcodes),
            udate or strftime("%Y-%m-%d"),
            utime or strftime("%H:%M:%S")
        ))
        if len(bucket) < BUCKET_SIZE:
            return
    if len(bucket) == 0:
        return
    cnx = cnxpool.get_connection()
    cursor = None
    stmt = """
        UPDATE secu.kline_d_b_lr_tags
        SET 
            tags = REPLACE('wcc_predict_ready', 'wcc_predict'),
            udate = %s,
            utime = %s,
        WHERE 
            code = %s
            AND klid = %s
            AND MATCH(tags) AGAINST('wcc_predict_ready')
    """
    try:
        cursor = cnx.cursor()
        cursor.executemany(WCC_INSERT, bucket)
        cursor.executemany(stmt, [(t[-2], t[-1], t[0], t[2]) for t in bucket])
        cnx.commit()
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        bucket = []
        if cnx.is_connected():
            cursor.close()
            cnx.close()


class SavePredictionCallback(tf.keras.callbacks.Callback):

    def on_predict_batch_end(self, batch, logs=None):
        tf.print('batch=', batch, ', logs=', logs,
                 ', history=', self.model.history)


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


def _setupTensorflow(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # interim workaround to fix memory leak issue
    tf.keras.backend.clear_session()
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        if args.gpu_grow_mem:
            try:
                print('{} enabling memory growth for {}'.format(
                    strftime("%H:%M:%S"), physical_devices[0]))
                tf.config.experimental.set_memory_growth(
                    physical_devices[0], True)
            except:
                print(
                    'Invalid device or cannot modify virtual devices once initialized.\n'
                    + sys.exc_info()[0])
                pass
        if args.limit_gpu_mem is not None:
            # Restrict TensorFlow to only allocate the specified memory on the first GPU
            try:
                print('{} setting GPU memory limit to {} MB'.format(
                    strftime("%H:%M:%S"), args.limit_gpu_mem*1024))
                tf.config.experimental.set_virtual_device_configuration(
                    physical_devices[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.limit_gpu_mem*1024)])
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(strftime("%H:%M:%S"), len(physical_devices),
                      "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

    if args.enable_xla:
        # enalbe XLA
        tf.config.optimizer.set_jit(True)
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_xla_devices'


@ray.remote
def _load_data(work, num_actors, min_rcode, shared_args, data_queue):
    global cnxpool
    host = shared_args['db_host']
    port = shared_args['db_port']
    pwd = shared_args['db_pwd']
    parallel = shared_args['args'].parallel

    if cnxpool is None:
        _init(1, host, port, pwd)

    actor_pool = ray.util.ActorPool(
        [ray.get_actor("DataLoader_{}".format(i)) for i in range(num_actors)]
    )

    # poll work request from 'work_queue' for data loading, and push to 'data_queue'
    def load_data(next_work):
        code, date_start, date_end, klid = next_work
        batch, rcodes = _process(actor_pool,
                                 code,
                                 klid,
                                 date_start,
                                 date_end,
                                 min_rcode,
                                 shared_args
                                 )
        if len(batch) < min_rcode or len(rcodes) < min_rcode:
            print('{} {}@({},{},{}) insufficient data for prediction. #batch: {}, #rcode: {}'.format(
                strftime("%H:%M:%S"), code, klid, date_start, date_end, len(batch), len(rcodes)))
            data_queue.put({'code': code, 'date': date_end,
                            'klid': klid, 'batch': None, 'rcodes': None})
            return
        data_queue.put({'code': code, 'date': date_end,
                        'klid': klid, 'batch': batch, 'rcodes': rcodes})

    c = 0
    elapsed = 0
    for next_item in work:
        if c >= 1000 and c < 2000:
            s = time.time()
            load_data(next_item)
            e = time.time()
            et = e-s
            elapsed += et
            print('{} load_data: {}'.format(
                strftime("%H:%M:%S"), et), file=sys.stderr)
        else:
            if c == 2000:
                print('{} load_data average: {}'.format(
                    strftime("%H:%M:%S"), elapsed/1000), file=sys.stderr)
            load_data(next_item)
        c += 1
        # sleep(0.1)

    for _ in range(parallel):
        data_queue.put('done')

    return True


@ray.remote
def _predict(model_path, max_batch_size, data_queue, infer_queue, args):
    _setupTensorflow(args)
    # poll work from 'data_queue', run inference, and push result to infer_queue
    model = _load_model(model_path)

    def predict():
        try:
            next_work = data_queue.get()

            if isinstance(next_work, str) and next_work == 'done':
                infer_queue.put('done')
                return True

            batch = next_work['batch']
            p = None
            if batch is not None and len(batch) > 0:
                p = model.predict(batch, batch_size=max_batch_size)
                p = np.squeeze(p)
            infer_queue.put(
                {'code': next_work['code'],
                 'date': next_work['date'],
                 'klid': next_work['klid'],
                 'result': p,
                 'rcodes': next_work['rcodes'],
                 'udate': strftime("%Y-%m-%d"),
                 'utime': strftime("%H:%M:%S")
                 }
            )
        except Exception:
            sleep(0.5)
            pass
        return False

    c = 0
    elapsed = 0
    done = False
    while not done:
        if data_queue.empty():
            sleep(0.5)
            continue

        if c >= 1000 and c < 2000:
            s = time.time()
            done = predict()
            e = time.time()
            et = e-s
            elapsed += et
            print('{} predict: {}'.format(
                strftime("%H:%M:%S"), et), file=sys.stderr)
        else:
            if c == 2000:
                print('{} predict average: {}'.format(
                    strftime("%H:%M:%S"), elapsed/1000), file=sys.stderr)
            done = predict()
        c += 1

    return done


def _tag_wcc_predict_insufficient(code, klid, udate, utime):
    global cnxpool
    cnx = cnxpool.get_connection()
    cursor = None
    stmt = """
        UPDATE secu.kline_d_b_lr_tags
        SET 
            tags = REPLACE('wcc_predict_ready', 'wcc_predict_insufficient'),
            udate = %s,
            utime = %s,
        WHERE 
            code = %s
            AND klid = %s
            AND MATCH(tags) AGAINST('wcc_predict_ready')
    """
    try:
        cursor = cnx.cursor()
        cursor.execute(stmt, (udate, utime, code, klid))
        cnx.commit()
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        if cnx.is_connected():
            cursor.close()
            cnx.close()


@ray.remote
def _save_infer_result(top_k, shared_args, infer_queue):
    global cnxpool
    db_host = shared_args['db_host']
    db_port = shared_args['db_port']
    db_pwd = shared_args['db_pwd']
    parallel = shared_args['args'].parallel

    if cnxpool is None:
        _init(1, db_host, db_port, db_pwd)

    def _inner_work():
        # poll work request from 'infer_queue' for saving inference result and handle persistence
        if infer_queue.empty():
            sleep(5)
            return 0
        try:
            next_result = infer_queue.get()

            if isinstance(next_result, str) and next_result == 'done':
                # flush bucket
                _save_prediction()
                return 1

            result = next_result['result']
            rcodes = next_result['rcodes']
            code = next_result['code']
            date = next_result['date']
            klid = next_result['klid']
            udate = next_result['udate']
            utime = next_result['utime']
            if result is None or len(result) == 0:
                _tag_wcc_predict_insufficient(code, klid, udate, utime)
            else:
                _save_prediction(code, klid, date, rcodes,
                                 top_k, result, udate, utime)
        except Exception:
            sleep(2)
            pass

        return 0

    done = 0
    while done < parallel:
        done += _inner_work()

    cnxpool._remove_connections()

    return done


def predict_wcc(num_actors, min_rcode, max_batch_size, model_path, top_k, shared_args):
    data_queue = Queue(maxsize=128)
    infer_queue = Queue(maxsize=128)
    db_host = shared_args['db_host']
    db_port = shared_args['db_port']
    db_pwd = shared_args['db_pwd']
    # anchors = shared_args['anchors']
    max_step = shared_args['max_step']
    time_shift = shared_args['time_shift']
    corl_prior = shared_args['corl_prior']
    args = shared_args['args']

    # actors will be retrieved by name in remote functions
    actors = [DataLoader.options(name='DataLoader_{}'.format(i)).remote(
        shared_args) for i in range(num_actors)]
    # ray.util.ActorPool(
    #     # [ray.get_actor("DataLoader_" + str(i)) for i in range(num_actors)]
    #     [DataLoader.options(name='DataLoader_{}'.format(i)).remote(
    #         shared_args) for i in range(num_actors)]
    # )

    work = getWorkloadForPredictionFromTags(
        corl_prior,
        max_step,
        time_shift,
        db_host,
        db_port,
        db_pwd)

    d = _load_data.remote(work,
                          num_actors,
                          min_rcode,
                          shared_args,
                          data_queue)

    s = _save_infer_result.remote(top_k,
                                  shared_args,
                                  infer_queue)

    # There're many unknown issues running GPU inference in ray worker...
    gpu_alloc = 1.0 / args.parallel
    p = []
    for i in range(args.parallel):
        p.append(_predict.options(num_gpus=gpu_alloc).remote(model_path,
                                                             max_batch_size,
                                                             data_queue,
                                                             infer_queue,
                                                             args)
                 )
        if i+1 < args.parallel:
            sleep(0.7)

    if ray.get(d) and ray.get(s) and all(r == True for r in list(ray.get(p))):
        print('{} inference completed. total workload: {}'.format(
            strftime("%H:%M:%S"), len(work)))
    else:
        print('{} inference completed with exception. total workload: {}'.format(
            strftime("%H:%M:%S"), len(work)))
