"""Create the input data pipeline using `tf.data`"""

from base import connect, getSeries, getBatch, ftQueryTpl, k_cols
from time import strftime
from joblib import Parallel, delayed
from loky import get_reusable_executor
import tensorflow as tf
import sys
import multiprocessing
import numpy as np

qk, qd = None, None
parallel = None
feat_cols = []
time_shift = None
max_step = None
_prefetch = None


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


def _getExecutor():
    global parallel, _executor, _prefetch
    if _executor is not None:
        return _executor
    _executor = get_reusable_executor(
        max_workers=parallel*_prefetch, timeout=20)
    return _executor


def _getSeries(p):
    uuid, code, klid, rcode, val, max_step, time_shift, ftQueryK, ftQueryD = p
    return getSeries(uuid, code, klid, rcode, val, max_step, time_shift, ftQueryK, ftQueryD)


def _loadTestSet(max_step, ntest):
    global parallel, time_shift
    cnx = connect()
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
        # print("====uuid:{},code:{},klid:{},rcode:{},val:{},max_step:{},time_shift:{},qk:{},qd:{}".format(
        #     tset[0][0],tset[0][1],tset[0][2],tset[0][3],tset[0][4],max_step,time_shift,qk,qd
        # ))
        r = Parallel(n_jobs=parallel)(delayed(getSeries)(
            uuid, code, klid, rcode, val, max_step, time_shift, qk, qd
        ) for uuid, code, klid, rcode, val in tset)
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
    global max_step, parallel, time_shift
    print("{} loading training set {}...".format(
        strftime("%H:%M:%S"), flag))
    cnx = connect()
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
            #joblib doesn't support nested threading
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
    cnx = connect()
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


def getInputs(start=0, shift=0, cols=None, step=30, cores=multiprocessing.cpu_count(), prefetch=2):
    """Input function for the wcc training dataset.

    Returns:
        A dictionary containing:
        uuids,features,labels,seqlens,train_iter,test_iter
    """
    # Create dataset for training
    global feat_cols, max_step, time_shift, parallel, _prefetch
    time_shift = shift
    feat_cols = cols
    max_step = step
    parallel = cores
    _prefetch = prefetch
    feat_size = len(cols)*2*(shift+1)
    print("{} Using parallel level:{}".format(strftime("%H:%M:%S"), parallel))
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
            )
        ).batch(1).prefetch(prefetch)
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
