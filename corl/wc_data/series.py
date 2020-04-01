# from __future__ import absolute_import, division, print_function, unicode_literals

from time import strftime
from mysql.connector.pooling import MySQLConnectionPool
import sys
import ray
import numpy as np

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


@ray.remote
def getSeries(code, klid, rcode, val, shared_args):
    # code, klid, rcode, val, max_step, time_shift, qk, qd = p
    max_step = shared_args['max_step']
    time_shift = shared_args['time_shift']
    s = max(0, klid - max_step + 1 - time_shift)
    batch, total = _getBatch(code, s, klid, rcode, shared_args)
    return batch, val, total


def _getBatch(code, s, e, rcode, shared_args):
    '''
    Returns:
    [max_step, feature*time_shift], length
    '''
    global cnxpool
    if cnxpool is None:
        db_host = shared_args['db_host']
        db_port = shared_args['db_port']
        db_pwd = shared_args['db_pwd']
        _init(1, db_host, db_port, db_pwd)

    max_step = shared_args['max_step']
    time_shift = shared_args['time_shift']
    qk = shared_args['qk']
    qd = shared_args['qd_idx'] if rcode in shared_args[
        'index_list'] else shared_args['qd']
    cnx = cnxpool.get_connection()
    fcursor = cnx.cursor(buffered=True)
    try:
        fcursor.execute(qk, (code, s, e, max_step + time_shift))
        col_names = fcursor.column_names
        featSize = (len(col_names) - 1) * 2
        total = fcursor.rowcount
        rows = fcursor.fetchall()
        # extract dates and transform to sql 'in' query
        dates = [r[0] for r in rows]
        dateStr = "'{}'".format("','".join(dates))
        ym = {d.replace('-', '')[:-2]
              for d in dates}  #extract year-month to a set
        ymStr = ",".join(ym)
        qd = qd.format(ymStr, dateStr)
        fcursor.execute(qd, (rcode, max_step + time_shift))
        rtotal = fcursor.rowcount
        r_rows = fcursor.fetchall()
        if total != rtotal:
            raise ValueError("{} prior data size {} != {}'s: {}".format(
                rcode, rtotal, code, total))
        batch = []
        for t in range(time_shift + 1):
            steps = np.zeros((max_step, featSize), dtype='f')
            offset = max_step + time_shift - total
            s = max(0, t - offset)
            e = total - time_shift + t
            for i, row in enumerate(rows[s:e]):
                for j, col in enumerate(row[1:]):
                    steps[i + offset][j] = col
            for i, row in enumerate(r_rows[s:e]):
                for j, col in enumerate(row[1:]):
                    steps[i + offset][j + featSize // 2] = col
            batch.append(steps)
        return np.concatenate(batch, 1), total - time_shift
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        fcursor.close()
        cnx.close()