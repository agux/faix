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


def _process(code, klid, date, min_rcode, shared_args):
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
    tasks = [getSeries_v2.remote(
        code, klid, rcode, None, shared_args) for rcode in rcodes]
    return np.array(ray.get(tasks), np.float32)



@ray.remote
def predict_wcc(work, min_rcode, model, shared_args):
    global cnxpool
    if cnxpool is None:
        db_host = shared_args['db_host']
        db_port = shared_args['db_port']
        db_pwd = shared_args['db_pwd']
        _init(1, db_host, db_port, db_pwd)
    for code, klid, date in work:
        batch = _process(code, klid, date, min_rcode, shared_args)
        #TODO use model to predict
        prediction = model.predict(batch)
        
