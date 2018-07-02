"""Create the input data pipeline using `tf.data`"""

from time import strftime
# from joblib import Parallel, delayed
from loky import get_reusable_executor
from google.cloud import bigquery as bq
import tensorflow as tf
import sys
import os
import multiprocessing
import numpy as np

PROJECT_ID = 'linen-mapper-187215'
MAX_BNO = 1000000
MAX_TEST_SET = 10
qk, qd = None, None
parallel = None
feat_cols = []
time_shift = None
max_step = None
_prefetch = None

k_cols = ["lr"]
fs_stats = {}

client = bq.Client(project=PROJECT_ID)

_executor = None

ftQueryTpl = (
    "SELECT  "
    "    date, "
    "    {0} "  # (d.COLS - mean) / std
    "FROM "
    "    secu.kline_d_b "
    "WHERE "
    "    code = @code "
    "    {1} "
    "ORDER BY klid "
    "LIMIT @limit "
)


def _getExecutor():
    global parallel, _executor, _prefetch
    if _executor is not None:
        return _executor
    _executor = get_reusable_executor(
        max_workers=parallel*_prefetch,
        # initializer=_init,
        # initargs=(db_pool_size, db_host, db_port, db_pwd),
        timeout=4500)
    return _executor


def _getBatch(code, s, e, rcode, max_step, time_shift, ftQueryK, ftQueryD):
    '''
    [max_step, feature*time_shift], length
    '''
    global client
    limit = max_step+time_shift
    query_params = [
        bq.ScalarQueryParameter('code', 'STRING', code),
        bq.ScalarQueryParameter('klid_start', 'INT64', s),
        bq.ScalarQueryParameter('klid_end', 'INT64', e),
        bq.ScalarQueryParameter('limit', 'INT64', limit)
    ]
    job_config = bq.QueryJobConfig()
    job_config.query_parameters = query_params
    query_job = client.query(
        ftQueryK,
        job_config=job_config)
    rows = list(query_job)
    total = len(rows)
    num_feats = len(rows[0])
    featSize = (num_feats-1)*2
    # extract dates and transform to sql 'in' query
    dates = "'{}'".format("','".join([r.date for r in rows]))
    qd = ftQueryD.format(dates)
    query_params = [
        bq.ScalarQueryParameter('code', 'STRING', rcode),
        bq.ScalarQueryParameter('limit', 'INT64', limit)
    ]
    job_config.query_parameters = query_params
    query_job = client.query(
        qd,
        job_config=job_config)
    r_rows = list(query_job)
    rtotal = len(r_rows)
    if total != rtotal:
        raise ValueError(
            "rcode({}) prior data size {} != code({}): {}".format(rcode, rtotal, code, total))
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


def _getSeries(p):
    code, klid, rcode, val, max_step, time_shift, ftQueryK, ftQueryD = p
    s = max(0, klid-max_step+1-time_shift)
    batch, total = _getBatch(
        code, s, klid, rcode, max_step, time_shift, ftQueryK, ftQueryD)
    return batch, val, total


def _loadTestSet(max_step, ntest, vset=None):
    global client, time_shift
    setno = vset or np.random.randint(ntest)
    flag = 'TEST_{}'.format(setno)
    print('{} selected test set: {}'.format(
        strftime("%H:%M:%S"), flag))
    query_params = [
        bq.ScalarQueryParameter('flag', 'STRING', flag),
    ]
    job_config = bq.QueryJobConfig()
    job_config.query_parameters = query_params
    query = (
        "SELECT "
        "   code, klid, rcode, corl_stz "
        "FROM "
        "   secu.wcc_trn "
        "WHERE "
        "   flag = @flag "
    )
    query_job = client.query(
        query,
        job_config=job_config)
    tset = list(query_job)
    qk, qd = _getFtQuery()
    exc = _getExecutor()
    params = [(code, klid, rcode, val, max_step, time_shift, qk, qd)
              for code, klid, rcode, val in tset]
    r = list(exc.map(_getSeries, params))
    data, vals, seqlen = zip(*r)
    # data = [batch, max_step, feature*time_shift]
    # vals = [batch]
    # seqlen = [batch]
    return np.array(data, 'f'), np.array(vals, 'f'), np.array(seqlen, 'i')


def _loadTrainingData(flag):
    global client, max_step, time_shift
    print("{} loading training set {}...".format(
        strftime("%H:%M:%S"), flag))
    query = (
        'SELECT '
        "   code, klid, rcode, corl_stz "
        'FROM '
        '   secu.wcc_trn '
        'WHERE '
        "   flag = @flag"
    )
    query_params = [
        bq.ScalarQueryParameter('flag', 'STRING', flag),
    ]
    job_config = bq.QueryJobConfig()
    job_config.query_parameters = query_params
    query_job = client.query(
        query,
        job_config=job_config
    )
    train_set = list(query_job)
    total = len(train_set)
    data, vals, seqlen = [], [], []
    if total > 0:
        qk, qd = _getFtQuery()
        # joblib doesn't support nested threading
        exc = _getExecutor()
        params = [(code, klid, rcode, val, max_step, time_shift, qk, qd)
                  for code, klid, rcode, val in train_set]
        r = list(exc.map(_getSeries, params))
        data, vals, seqlen = zip(*r)
    # data = [batch, max_step, feature*time_shift]
    # vals = [batch]
    # seqlen = [batch]
    return np.array(data, 'f'), np.array(vals, 'f'), np.array(seqlen, 'i')


def _getFtQuery():
    global qk, qd, feat_cols, fs_stats
    if qk is not None and qd is not None:
        return qk, qd

    k_cols = feat_cols
    sep = " "
    p_kline = sep.join(
        ["({0}-{1})/{2} {0},".format(c, fs_stats["{}_mean".format(c)], fs_stats["{}_std".format(c)]) for c in k_cols])
    p_kline = p_kline[:-1]  # strip last comma

    qk = ftQueryTpl.format(
        p_kline, " AND klid BETWEEN @klid_start AND @klid_end ")
    qd = ftQueryTpl.format(p_kline, " AND date in ({})")
    print('qk:\n{}'.format(qk))
    print('qd:\n{}'.format(qd))
    return qk, qd


def init_fs_stats():
    global feat_cols, fs_stats
    query = (
        'select fields, mean, std from secu.fs_stats '
        "where method = 'standardization'"
        "and fields in ({})"
    ).format("'{}'".format("','".join(feat_cols)))
    query_job = client.query(query)
    for r in query_job:
        print('{} mean={}, std={}'.format(r.fields, r.mean, r.std))
        fs_stats['{}_mean'.format(r.fields)] = r.mean
        fs_stats['{}_std'.format(r.fields)] = r.std


def getInputs(start=0, shift=0, cols=None, step=30, test_batch_size=None,
              cores=multiprocessing.cpu_count(), pfetch=2, vset=None):
    """Input function for the wcc training dataset.

    Returns:
        A dictionary containing:
        features,labels,seqlens,train_iter,test_iter
    """
    # Create dataset for training
    global feat_cols, max_step, time_shift, parallel, _prefetch, client
    time_shift = shift
    feat_cols = cols or k_cols
    max_step = step
    parallel = cores
    _prefetch = pfetch
    feat_size = len(feat_cols)*2*(shift+1)
    print("{} Using parallel: {}, prefetch: {}".format(
        strftime("%H:%M:%S"), parallel, _prefetch))
    init_fs_stats()
    with tf.variable_scope("build_inputs"):
        # query max flag from wcc_trn and fill a slice with flags between start and max
        flags = ["TRAIN_{}".format(bno) for bno in range(start, MAX_BNO)]
        train_dataset = tf.data.Dataset.from_tensor_slices(flags).map(
            lambda f: tuple(
                tf.py_func(_loadTrainingData, [f], [
                           tf.float32, tf.float32, tf.int32])
            ), _prefetch
        ).batch(1).prefetch(_prefetch)
        # Create dataset for testing
        test_dataset = tf.data.Dataset.from_tensor_slices(
            _loadTestSet(step, MAX_TEST_SET+1, vset)).batch(test_batch_size).repeat()

        train_iterator = train_dataset.make_one_shot_iterator()
        test_iterator = test_dataset.make_one_shot_iterator()

        handle = tf.placeholder(tf.string, shape=[])
        types = (tf.float32, tf.float32, tf.int32)
        shapes = (tf.TensorShape([None, step, feat_size]),
                  tf.TensorShape([None]), tf.TensorShape([None]))
        iter = tf.data.Iterator.from_string_handle(
            handle, types, train_dataset.output_shapes)

        next_el = iter.get_next()
        features = tf.squeeze(next_el[0])
        features.set_shape(shapes[0])
        labels = tf.squeeze(next_el[1])
        labels.set_shape(shapes[1])
        seqlens = tf.squeeze(next_el[2])
        seqlens.set_shape(shapes[2])
        print("{} features:{}, labels:{}, seqlens:{}".format(
            strftime("%H:%M:%S"), features.get_shape(), labels.get_shape(), seqlens.get_shape()))
        # return a dictionary
        inputs = {'features': features, 'labels': labels,
                  'seqlens': seqlens, 'handle': handle,
                  'train_iter': train_iterator, 'test_iter': test_iterator}
        return inputs
