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

qk, qd = None, None
parallel = None
feat_cols = []
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
    "    {0} "  # (d.COLS - mean) / std
    "FROM "
    "    kline_d_b d "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        %s code, "
    "        t.method, "
    "        {1} "  # mean & std fields
    "    FROM "
    "        fs_stats t "
    "    WHERE "
    "        t.method = 'standardization' "
    "    GROUP BY code , t.method) s USING (code) "
    "WHERE "
    "    d.code = %s "
    "    {2} "
    "ORDER BY klid "
    "LIMIT %s "
)

maxbno_query = (
    "SELECT  "
    "    MAX(CONVERT( SUBSTRING_INDEX(flag, '_', - 1) , UNSIGNED INTEGER)) AS max_bno "
    "FROM "
    "    (SELECT DISTINCT "
    "        flag "
    "    FROM "
    "        wcc_trn "
    "    WHERE "
    "        flag LIKE %s) t "
)

_executor = None

cnxpool = None


def _init():
    global cnxpool, db_pool_size, db_host, db_port, db_pwd
    print("PID %d: initializing mysql connection pool..." % os.getpid())
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
        connect_timeout=60000
    )


def _getExecutor():
    global parallel, _executor, _prefetch
    if _executor is not None:
        return _executor
    _executor = get_reusable_executor(
        max_workers=parallel*_prefetch,
        initializer=_init,
        timeout=45)
    return _executor


def _getBatch(code, s, e, rcode, max_step, time_shift, ftQueryK, ftQueryD):
    '''
    [max_step, feature*time_shift], length
    '''
    global cnxpool
    cnx = cnxpool.get_connection()
    fcursor = cnx.cursor(buffered=True)
    try:
        fcursor.execute(ftQueryK, (code, code, s, e, max_step+time_shift))
        col_names = fcursor.column_names
        featSize = (len(col_names)-1)*2
        total = fcursor.rowcount
        rows = fcursor.fetchall()
        # extract dates and transform to sql 'in' query
        dates = "'{}'".format("','".join([r[0] for r in rows]))
        qd = ftQueryD.format(dates)
        fcursor.execute(qd, (rcode, rcode, max_step+time_shift))
        rtotal = fcursor.rowcount
        r_rows = fcursor.fetchall()
        if total != rtotal:
            raise ValueError(
                "rcode prior data size {} != code's: {}".format(rtotal, total))
        batch = []
        for t in range(time_shift+1):
            steps = np.zeros((max_step, featSize), dtype='f')
            offset = max_step + time_shift - total
            s = max(0, t - offset)
            e = total - time_shift + t
            for i, row in enumerate(rows[s:e]):
                for j, col in enumerate(row[1:]):
                    steps[i+offset][j] = col
            for i, row in enumerate(r_rows[s:e]):
                for j, col in enumerate(row[1:]):
                    steps[i+offset][j+featSize//2] = col
            batch.append(steps)
        return np.concatenate(batch, 1), total - time_shift
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        fcursor.close()
        cnx.close()


def _getSeries(p):
    uuid, code, klid, rcode, val, max_step, time_shift, ftQueryK, ftQueryD = p
    s = max(0, klid-max_step+1-time_shift)
    batch, total = _getBatch(
        code, s, klid, rcode, max_step, time_shift, ftQueryK, ftQueryD)
    return uuid, batch, val, total


def _loadTestSet(max_step, ntest):
    global parallel, time_shift, cnxpool
    cnx = cnxpool.get_connection()
    try:
        rn = np.random.randint(ntest)
        flag = 'TEST_{}'.format(rn)
        print('{} selected test set: {}'.format(
            strftime("%H:%M:%S"), flag))
        query = (
            "SELECT "
            "   uuid, code, klid, rcode, corl "
            "FROM "
            "   wcc_trn "
            "WHERE "
            "   flag = %s "
        )
        cursor = cnx.cursor(buffered=True)
        cursor.execute(query, (flag,))
        tset = cursor.fetchall()
        cursor.close()
        qk, qd = _getFtQuery()
        exc = _getExecutor()
        params = [(uuid, code, klid, rcode, val, max_step, time_shift, qk, qd)
                  for uuid, code, klid, rcode, val in tset]
        r = list(exc.map(_getSeries, params))
        uuids, data, vals, seqlen = zip(*r)
        # data = [batch, max_step, feature*time_shift]
        # vals = [batch]
        # seqlen = [batch]
        return np.array(uuids, 'U'), np.array(data, 'f'), np.array(vals, 'f'), np.array(seqlen, 'i')
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


def _loadTrainingData(flag):
    global max_step, parallel, time_shift, cnxpool
    print("{} loading training set {}...".format(
        strftime("%H:%M:%S"), flag))
    cnx = cnxpool.get_connection()
    try:
        cursor = cnx.cursor(buffered=True)
        query = (
            'SELECT '
            "   uuid, code, klid, rcode, corl "
            'FROM '
            '   wcc_trn '
            'WHERE '
            "   flag = %s"
        )
        cursor.execute(query, (flag,))
        train_set = cursor.fetchall()
        total = cursor.rowcount
        cursor.close()
        uuids, data, vals, seqlen = [], [], [], []
        if total > 0:
            qk, qd = _getFtQuery()
            # joblib doesn't support nested threading
            exc = _getExecutor()
            params = [(uuid, code, klid, rcode, val, max_step, time_shift, qk, qd)
                      for uuid, code, klid, rcode, val in train_set]
            r = list(exc.map(_getSeries, params))
            uuids, data, vals, seqlen = zip(*r)
        # data = [batch, max_step, feature*time_shift]
        # vals = [batch]
        # seqlen = [batch]
        return np.array(uuids, 'U'), np.array(data, 'f'), np.array(vals, 'f'), np.array(seqlen, 'i')
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


def _getFtQuery():
    global qk, qd, feat_cols
    if qk is not None and qd is not None:
        return qk, qd

    k_cols = feat_cols
    sep = " "
    p_kline = sep.join(
        ["(d.{0}-s.{0}_mean)/s.{0}_std {0},".format(c) for c in k_cols])
    p_kline = p_kline[:-1]  # strip last comma
    stats_tpl = (
        " MAX(CASE "
        "     WHEN t.fields = '{0}' THEN t.mean "
        "     ELSE NULL "
        " END) AS {0}_mean, "
        " MAX(CASE "
        "     WHEN t.fields = '{0}' THEN t.std "
        "     ELSE NULL "
        " END) AS {0}_std,"
    )
    stats = sep.join([stats_tpl.format(c) for c in k_cols])
    stats = stats[:-1]  # strip last comma

    qk = ftQueryTpl.format(p_kline, stats,
                           " AND d.klid BETWEEN %s AND %s ")
    qd = ftQueryTpl.format(p_kline, stats,
                           " AND d.date in ({})")
    return qk, qd


def _getDataSetMeta(flag, start=0):
    global cnxpool
    cnx = cnxpool.get_connection()
    max_bno, batch_size = None, None
    try:
        print('{} querying max batch no for {} set...'.format(
            strftime("%H:%M:%S"), flag))
        cursor = cnx.cursor()
        cursor.execute(maxbno_query, ("{}_%".format(flag),))
        row = cursor.fetchone()
        max_bno = row[0]
        print('{} start: {}, max batch no: {}'.format(
            strftime("%H:%M:%S"), start, max_bno))
        if start > max_bno:
            print('{} no more material to {}.'.format(
                strftime("%H:%M:%S"), flag.lower()))
            return None, None
        query = (
            "SELECT  "
            "    COUNT(*) "
            "FROM "
            "    wcc_trn "
            "WHERE "
            "    flag = %s "
        )
        cursor.execute(query, ("{}_{}".format(flag, start),))
        row = cursor.fetchone()
        batch_size = row[0]
        print('{} batch size: {}'.format(strftime("%H:%M:%S"), batch_size))
        if batch_size == 0:
            print('{} no more material to {}.'.format(
                strftime("%H:%M:%S"), flag.lower()))
            return None, None
        cursor.close()
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()
    return max_bno, batch_size


def getInputs(start=0, shift=0, cols=None, step=30, cores=multiprocessing.cpu_count(), pfetch=2, pool=None, host=None, port=None, pwd=None):
    """Input function for the wcc training dataset.

    Returns:
        A dictionary containing:
        uuids,features,labels,seqlens,train_iter,test_iter
    """
    # Create dataset for training
    global feat_cols, max_step, time_shift, parallel, _prefetch, db_pool_size, db_host, db_port, db_pwd
    time_shift = shift
    feat_cols = cols
    max_step = step
    parallel = cores
    _prefetch = pfetch
    db_pool_size = pool
    feat_size = len(cols)*2*(shift+1)
    db_host = host
    db_port = port
    db_pwd = pwd
    print("{} Using parallel: {}, prefetch: {} db_host: {} port: {} pwd: {}".format(
        strftime("%H:%M:%S"), parallel, _prefetch, db_host, db_port, db_pwd))
    _init()
    with tf.variable_scope("build_inputs"):
        # query max flag from wcc_trn and fill a slice with flags between start and max
        max_bno, _ = _getDataSetMeta("TRAIN", start)
        if max_bno is None:
            return None
        flags = ["TRAIN_{}".format(bno) for bno in range(start, max_bno+1)]
        train_dataset = tf.data.Dataset.from_tensor_slices(flags).map(
            lambda f: tuple(
                tf.py_func(_loadTrainingData, [f], [
                    tf.string, tf.float32, tf.float32, tf.int32])
            ), parallel
        ).batch(1).prefetch(_prefetch)
        # Create dataset for testing
        max_bno, batch_size = _getDataSetMeta("TEST", 1)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            _loadTestSet(step, max_bno+1)).batch(batch_size).repeat()

        train_iterator = train_dataset.make_one_shot_iterator()
        test_iterator = test_dataset.make_one_shot_iterator()

        handle = tf.placeholder(tf.string, shape=[])
        types = (tf.string, tf.float32, tf.float32, tf.int32)
        shapes = (tf.TensorShape([None]), tf.TensorShape(
            [None, step, feat_size]), tf.TensorShape([None]), tf.TensorShape([None]))
        iter = tf.data.Iterator.from_string_handle(
            handle, types, train_dataset.output_shapes)

        next_el = iter.get_next()
        uuids = tf.squeeze(next_el[0])
        uuids.set_shape(shapes[0])
        features = tf.squeeze(next_el[1])
        features.set_shape(shapes[1])
        labels = tf.squeeze(next_el[2])
        labels.set_shape(shapes[2])
        seqlens = tf.squeeze(next_el[3])
        seqlens.set_shape(shapes[3])
        print("{} uuids:{}, features:{}, labels:{}, seqlens:{}".format(
            strftime("%H:%M:%S"), uuids.get_shape(), features.get_shape(), labels.get_shape(), seqlens.get_shape()))
        # return a dictionary
        inputs = {'uuids': uuids, 'features': features, 'labels': labels,
                  'seqlens': seqlens, 'handle': handle, 'train_iter': train_iterator, 'test_iter': test_iterator}
        return inputs
