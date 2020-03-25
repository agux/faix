"""Create the input data pipeline using `tf.data`"""

from time import strftime
# from joblib import Parallel, delayed
from loky import get_reusable_executor
from mysql.connector.pooling import MySQLConnectionPool
import tensorflow as tf
import sys
import os
import multiprocessing
import numpy as np

qk, qd, qd_idx = None, None, None
parallel = None
feat_cols = []
idxlst = None
time_shift = None
max_step = None
_prefetch = None
db_pool_size = None
db_host = None
db_port = None
db_pwd = None

k_cols = ["lr"]

ftQueryTpl = (
    "SELECT  "
    "    date, "
    "    {0} "  # fields
    "FROM "
    "    {1} d "  # table
    "WHERE "
    "    d.code = %s "
    "    {2} "
    "ORDER BY klid "
    "LIMIT %s ")

maxbno_query = ("SELECT  "
                "    MAX(bno) AS max_bno "
                "FROM "
                "    wcc_trn "
                "WHERE "
                "    flag = %s")

_executor = None

cnxpool = None


def _init(db_pool_size=None, db_host=None, db_port=None, db_pwd=None):
    global cnxpool
    print("PID {}: initializing mysql connection pool...".format(os.getpid()))
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
        connect_timeout=60000)


def _getExecutor():
    global parallel, _executor, _prefetch, db_pool_size, db_host, db_port, db_pwd
    if _executor is not None:
        return _executor
    _executor = get_reusable_executor(max_workers=parallel * _prefetch,
                                      initializer=_init,
                                      initargs=(db_pool_size, db_host, db_port,
                                                db_pwd),
                                      timeout=450)
    return _executor


def _getBatch(code, s, e, rcode, max_step, time_shift, qk, qd):
    '''
    [max_step, feature*time_shift], length
    '''
    global cnxpool
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
        ym = {d[:-2] for d in dates}  #extract year-month to a set
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


def _getSeries(p):
    code, klid, rcode, val, max_step, time_shift, qk, qd = p
    s = max(0, klid - max_step + 1 - time_shift)
    batch, total = _getBatch(code, s, klid, rcode, max_step, time_shift, qk,
                             qd)
    return batch, val, total


def _loadTestSet(max_step, ntest, vset=None):
    global parallel, time_shift, cnxpool
    cnx = cnxpool.get_connection()
    idxlst = _getIndex()
    try:
        setno = vset or np.random.randint(ntest)
        flag = 'TS'
        print('{} selected test set: {}'.format(strftime("%H:%M:%S"), setno))
        query = ("SELECT "
                 "   code, klid, rcode, corl_stz "
                 "FROM "
                 "   wcc_trn "
                 "WHERE "
                 "   flag = %s "
                 "   AND bno = %s")
        cursor = cnx.cursor(buffered=True)
        cursor.execute(query, (flag, setno))
        tset = cursor.fetchall()
        cursor.close()
        qk, qd, qd_idx = _getFtQuery()
        exc = _getExecutor()
        params = [(code, klid, rcode, val, max_step, time_shift, qk,
                   qd_idx if rcode in idxlst else qd)
                  for code, klid, rcode, val in tset]
        r = list(exc.map(_getSeries, params))
        data, vals, seqlen = zip(*r)
        # data = [batch, max_step, feature*time_shift]
        # vals = [batch]
        # seqlen = [batch]
        return {
            'features': np.array(data, 'f'),
            'seqlens': np.array(seqlen, 'i')
        }, np.array(vals, 'f'),
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


def _getIndex():
    '''
    Returns a set of index codes from idxlst table.
    '''
    global idxlst
    if idxlst is not None:
        return idxlst
    print("{} loading index...".format(strftime("%H:%M:%S")))
    cnx = cnxpool.get_connection()
    try:
        cursor = cnx.cursor(buffered=True)
        query = ('SELECT distinct code FROM idxlst')
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        idxlst = {c[0] for c in rows}
        return idxlst
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


def _loadTrainingData(flag, bno):
    global max_step, parallel, time_shift, cnxpool
    idxlst = _getIndex()
    print("{} loading training set {} {}...".format(strftime("%H:%M:%S"), flag, bno))
    cnx = cnxpool.get_connection()
    try:
        cursor = cnx.cursor(buffered=True)
        query = ('SELECT '
                 "   code, klid, rcode, corl_stz "
                 'FROM '
                 '   wcc_trn '
                 'WHERE '
                 "   flag = %s "
                 "   AND bno = %s")
        cursor.execute(query, (
            flag,
            bno,
        ))
        train_set = cursor.fetchall()
        total = cursor.rowcount
        cursor.close()
        data, vals, seqlen = [], [], []
        if total > 0:
            qk, qd, qd_idx = _getFtQuery()
            # joblib doesn't support nested threading
            exc = _getExecutor()
            params = [(code, klid, rcode, val, max_step, time_shift, qk,
                       qd_idx if rcode in idxlst else qd)
                      for code, klid, rcode, val in train_set]
            r = list(exc.map(_getSeries, params))
            data, vals, seqlen = zip(*r)
        # data = [batch, max_step, feature*time_shift]
        # seqlen = [batch]
        # vals = [batch]
        return {
            'features': np.array(data, 'f'),
            'seqlens': np.array(seqlen, 'i')
        }, np.array(vals, 'f')
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


def _getStats():
    '''
    Get statistics from fs_stats table.
    Returns:
    A dictionary of {"table:column" : tuple(mean, std)}
    '''
    global feat_cols
    print("{} loading statistics for {}...".format(strftime("%H:%M:%S"),
                                                   feat_cols))
    cnx = cnxpool.get_connection()
    try:
        cursor = cnx.cursor(buffered=True)
        query = ('SELECT '
                 "   `tab`, `fields`, `mean`, `std`"
                 'FROM '
                 '   fs_stats '
                 'WHERE '
                 "   method = 'standardization' "
                 "   AND tab in ('index_d_n_lr','kline_d_b_lr') "
                 "   AND fields in ({}) ")
        fields = "'{}'".format("','".join(feat_cols))
        query = query.format(fields)
        cursor.execute(query)
        rows = cursor.fetchall()
        total = cursor.rowcount
        cursor.close()
        stats = {}
        for t, f, m, s in rows:
            stats[('{}:{}'.format(t, f))] = (m, s)
        return stats
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


def _getFtQuery():
    global qk, qd, qd_idx, feat_cols
    if qk is not None and qd is not None and qd_idx is not None:
        return qk, qd, qd_idx

    stats = _getStats()
    k_cols = feat_cols
    p_kline = ""
    p_index = ""
    for k, v in feat_cols.items:
        if k.startswith('kline'):
            _, c = k.split(':')
            p_kline += "(d.{0}-{1})/{2} {0},".format(c, v[0], v[1])
        else:
            _, c = k.split(':')
            p_index += "(d.{0}-{1})/{2} {0},".format(c, v[0], v[1])
    p_kline = p_kline[:-1]  # strip last comma
    p_index = p_index[:-1]  # strip last comma
    qk = ftQueryTpl.format(p_kline, 'kline_d_b_lr',
                           " AND d.klid BETWEEN %s AND %s ")
    qd = ftQueryTpl.format(p_kline, 'kline_d_b_lr',
                           " AND d.ym in ({}) AND d.date in ({})")
    qd_idx = ftQueryTpl.format(p_index, 'index_d_n_lr',
                               " AND d.ym in ({}) AND d.date in ({})")
    return qk, qd, qd_idx


def _getDataSetMeta(flag):
    global cnxpool
    cnx = cnxpool.get_connection()
    max_bno, batch_size = None, None
    try:
        print('{} querying max batch no for {} set...'.format(
            strftime("%H:%M:%S"), flag))
        cursor = cnx.cursor()
        cursor.execute(maxbno_query, (flag, ))
        row = cursor.fetchone()
        max_bno = row[0]
        print('{} max batch no: {}'.format(strftime("%H:%M:%S"), max_bno))
        query = ("SELECT  "
                 "    COUNT(*) "
                 "FROM "
                 "    wcc_trn "
                 "WHERE "
                 "    flag = %s "
                 "    AND bno = 1 ")
        cursor.execute(query, (flag,))
        row = cursor.fetchone()
        batch_size = row[0]
        print('{} batch size: {}'.format(strftime("%H:%M:%S"), batch_size))
        if batch_size == 0:
            print('{} no more data for {}.'.format(strftime("%H:%M:%S"),
                                                   flag.lower()))
            return None, None
        cursor.close()
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()
    return max_bno, batch_size


def getInputs(shift=0,
              cols=None,
              step=30,
              cores=multiprocessing.cpu_count(),
              pfetch=2,
              pool=None,
              host=None,
              port=None,
              pwd=None,
              vset=None):
    """Input function for the wcc training dataset.

    Returns:
        A tuple of (inputs, targets). 
        Where inputs is a dictionary of {features,seqlens}.
        In summary, ({features, seqlens}, targets)
    """
    # Create dataset for training
    global feat_cols, max_step, time_shift, parallel, _prefetch, db_pool_size, db_host, db_port, db_pwd
    time_shift = shift
    feat_cols = cols or k_cols
    max_step = step
    parallel = cores
    _prefetch = pfetch
    db_pool_size = pool
    feat_size = len(feat_cols) * 2 * (shift + 1)
    db_host = host
    db_port = port
    db_pwd = pwd
    print("{} Using parallel: {}, prefetch: {} db_host: {} port: {}".format(
        strftime("%H:%M:%S"), parallel, _prefetch, db_host, db_port))
    _init(db_pool_size, db_host, db_port, db_pwd)

    # query max flag from wcc_trn and fill a slice with flags between start and max
    max_bno, _ = _getDataSetMeta("TR")
    if max_bno is None:
        return None
    bnums = [bno for bno in range(1, max_bno + 1)]
    ds_train = tf.data.Dataset.from_tensor_slices(bnums).map(
        lambda bno: tuple(
            tf.py_function(func=_loadTrainingData,
                           inp=['TR', bno],
                           Tout=[{
                               'features': tf.float32,
                               'seqlens': tf.int32
                           }, tf.float32])),
        _prefetch).batch(1).prefetch(_prefetch)
    # Create dataset for testing
    max_bno, batch_size = _getDataSetMeta("TS")
    ds_test = tf.data.Dataset.from_tensor_slices(
        _loadTestSet(step, max_bno + 1, vset)).batch(batch_size).repeat()

    return ds_train, ds_test
