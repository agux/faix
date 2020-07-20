import ray
import sys
import psutil
import os

import numpy as np
import tensorflow as tf

from time import strftime
from pathlib import Path
from mysql.connector.pooling import MySQLConnectionPool
from corl.wc_data.series import getSeries_v2

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
    global cnxpool
    print("{} initializing mysql connection pool...".format(
        strftime("%H:%M:%S")))
    cnxpool = MySQLConnectionPool(
        pool_name="dbpool",
        pool_size=db_pool_size or 5,
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
        dates = [r[0] for r in rows]
        # get rcodes from kline table
        rcodes_k = _get_rcodes_for(code, 'kline_d_b_lr', dates)
        # get rcodes from index table
        rcodes_i = _get_rcodes_for(code, 'index_d_n_lr', dates)
        return rcodes_k + rcodes_i
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        if cursor is not None:
            cursor.close()
        cnx.close()


def _process(code, klid, date, min_rcode, shared_args, shared_args_oid):
    # find eligible rcodes
    rcodes = _get_rcodes(
        code, klid, shared_args['max_step'], shared_args['time_shift'])
    # check rcodes
    if len(rcodes) < min_rcode:
        print('{} {}@({},{}) has {} eligible reference codes, skipping'.format(
            strftime("%H:%M:%S"),
            code, klid, date, len(rcodes),
        ))
        return []
    # retrive objectID for shared_sargs and pass to getSeries_v2
    tasks = [getSeries_v2.remote(
        code, klid, rcode, None, shared_args_oid) for rcode in rcodes]
    return np.array(ray.get(tasks), np.float32), np.array(rcodes, object)


def _save_prediction(code=None, klid=None, date=None, rcodes=None, top_k=None, predictions=None):
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
            *top_k_corl,
            *bottom_k_code,
            *bottom_k_corl,
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
        if cnx.is_connected():
            cursor.close()
            cnx.close()

@ray.remote
def predict_wcc(work, min_rcode, model, top_k, shared_args, shared_args_oid):
    global cnxpool
    if cnxpool is None:
        db_host = shared_args['db_host']
        db_port = shared_args['db_port']
        db_pwd = shared_args['db_pwd']
        _init(1, db_host, db_port, db_pwd)
    for code, klid, date in work:
        batch, rcodes = _process(code, klid, date, min_rcode, shared_args, shared_args_oid)
        if len(batch) < min_rcode or len(rcodes) < min_rcode:
            print('{} {}@({},{}) insufficient data for prediction. #batch: {}, #rcode: {}'.format(
                strftime("%H:%M:%S"), code, klid, date, len(batch), len(rcodes)))
            continue
        # use model to predict
        p = model.predict(batch)
        _save_prediction(code, klid, date, rcodes, top_k, p)
    # flush bucket
    _save_prediction()
