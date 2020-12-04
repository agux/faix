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
from corl.wc_data.input_fn import getWorkloadForPrediction

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
    # ray.init(
    #     num_cpus=psutil.cpu_count(logical=False),
    #     webui_host='127.0.0.1',  # TODO need a different port?
    #     memory=2 * 1024 * 1024 * 1024,  # 2G
    #     object_store_memory=512 * 1024 * 1024,  # 512M
    #     driver_object_store_memory=256 * 1024 * 1024    # 256M
    # )


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


def _process(pool, code, klid, date, min_rcode, shared_args, shared_args_oid):
    # find eligible rcodes
    rcodes = _get_rcodes(
        code, klid, shared_args['max_step'], shared_args['time_shift'])
    # check rcodes
    if len(rcodes) < min_rcode:
        print('{} {}@({},{}) has {} eligible reference codes, skipping'.format(
            strftime("%H:%M:%S"),
            code, klid, date, len(rcodes),
        ))
        return [], []
    # retrieve objectID for shared_sargs and pass to getSeries_v2
    # tasks = [getSeries_v2.remote(
    #     code, klid, rcode, None, shared_args_oid) for rcode in rcodes]
    tasks = pool.map(lambda a, rcode: a.get_series.remote(
        code, klid, rcode, None, shared_args_oid),
        rcodes
    )
    return np.array(list(tasks), np.float32), np.array(rcodes, object)


def _save_prediction(code=None, klid=None, date=None, rcodes=None, top_k=None, predictions=None):
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
            strftime("%Y-%m-%d"),
            strftime("%H:%M:%S")
        ))
        if len(bucket) < BUCKET_SIZE:
            return
    if len(bucket) == 0:
        return
    cnx = cnxpool.get_connection()
    cursor = None
    try:
        cursor = cnx.cursor()
        cursor.executemany(WCC_INSERT, bucket)
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


@ray.remote
def _load_data(work, num_actors, min_rcode, shared_args, shared_args_oid, data_queue):
    global cnxpool
    host = shared_args['db_host']
    port = shared_args['db_port']
    pwd = shared_args['db_pwd']
    if cnxpool is None:
        _init(1, host, port, pwd)

    actor_pool = ray.util.ActorPool(
        # [ray.get_actor("DataLoader_" + str(i)) for i in range(num_actors)]
        [DataLoader.remote(host=host, port=port, pwd=pwd)
         for _ in range(num_actors)]
    )

    # poll work request from 'work_queue' for data loading, and push to 'data_queue'
    def load_data(next_work):
        code, date, klid = next_work
        batch, rcodes = _process(actor_pool,
                                 code,
                                 klid,
                                 date,
                                 min_rcode,
                                 shared_args,
                                 shared_args_oid)
        if len(batch) < min_rcode or len(rcodes) < min_rcode:
            print('{} {}@({},{}) insufficient data for prediction. #batch: {}, #rcode: {}'.format(
                strftime("%H:%M:%S"), code, klid, date, len(batch), len(rcodes)))
            return
        data_queue.put({'code': code, 'date': date,
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

    return True


def _predict(model_path, max_batch_size, data_queue, infer_queue):
    # os.environ('CUDA_VISIBLE_DEVICES') = '0'
    # poll work from 'data_queue', run inference, and push result to infer_queue
    model = _load_model(model_path)

    def predict():
        try:
            next_work = data_queue.get()
            if isinstance(next_work, str) and next_work == 'done':
                if data_queue.empty():
                    infer_queue.put('done')
                    done = True
                else:
                    print('{} warning, data_queue is still not empty when ''done'' signal is received. qsize: {}'.format(
                        strftime("%H:%M:%S"), data_queue.size()))
                    data_queue.put_nowait('done')
                    return
            batch = next_work['batch']
            p = model.predict(batch, batch_size=max_batch_size)
            p = np.squeeze(p)
            infer_queue.put(
                {'code': next_work['code'],
                 'date': next_work['date'],
                 'klid': next_work['klid'],
                 'result': p,
                 'rcodes': next_work['rcodes']}
            )
        except Exception:
            sleep(0.5)
            pass

    c = 0
    elapsed = 0
    done = False
    while not done:
        if data_queue.empty():
            sleep(0.5)
            continue

        if c >= 1000 and c < 2000:
            s = time.time()
            predict()
            e = time.time()
            et = e-s
            elapsed += et
            print('{} predict: {}'.format(
                strftime("%H:%M:%S"), et), file=sys.stderr)
        else:
            if c == 2000:
                print('{} predict average: {}'.format(
                    strftime("%H:%M:%S"), elapsed/1000), file=sys.stderr)
            predict()
        c += 1
    return done


@ray.remote
def _save_infer_result(top_k, shared_args, infer_queue):
    global cnxpool
    db_host = shared_args['db_host']
    db_port = shared_args['db_port']
    db_pwd = shared_args['db_pwd']
    if cnxpool is None:
        _init(1, db_host, db_port, db_pwd)

    def _inner_work():
        # poll work request from 'infer_queue' for saving inference result and handle persistence
        if infer_queue.empty():
            sleep(5)
            return False
        try:
            next_result = infer_queue.get()
            if isinstance(next_result, str) and next_result == 'done':
                if infer_queue.empty():
                    # flush bucket
                    _save_prediction()
                    return True
                else:
                    print('{} warning, infer_queue is still not empty when ''done'' signal is received. qsize: {}'.format(
                        strftime("%H:%M:%S"), infer_queue.size()))
                    infer_queue.put_nowait('done')
            else:
                result = next_result['result']
                rcodes = next_result['rcodes']
                code = next_result['code']
                date = next_result['date']
                klid = next_result['klid']
                _save_prediction(code, klid, date, rcodes, top_k, result)
        except Exception:
            pass
        sleep(2)
        return False

    done = False
    while not done:
        done = _inner_work()

    cnxpool._remove_connections()

    return done


def predict_wcc(anchor, num_actors, corl_prior, min_rcode, max_batch_size, model_path, top_k, shared_args, shared_args_oid):
    data_queue = Queue(maxsize=16)
    infer_queue = Queue(maxsize=16)
    db_host = shared_args['db_host']
    db_port = shared_args['db_port']
    db_pwd = shared_args['db_pwd']
    anchors = shared_args['anchors']
    start_anchor = None if anchor == 0 else anchors[anchor-1]
    stop_anchor = None if anchor == len(anchors) else anchors[anchor]
    work = getWorkloadForPrediction(start_anchor, stop_anchor,
                                    corl_prior, db_host, db_port, db_pwd)

    d = _load_data.remote(work, num_actors, min_rcode, shared_args,
                          shared_args_oid, data_queue)
    s = _save_infer_result.remote(top_k, shared_args, infer_queue)

    # prediction using GPU will be running in master process. There're many unknown issues running in ray worker
    p = _predict(model_path, max_batch_size, data_queue, infer_queue)

    if d and s and p:
        print('{} inference completed. total workload: {}'.format(
            strftime("%H:%M:%S"), len(work)))
