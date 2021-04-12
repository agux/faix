

import os
import sys
import psutil
import ray

import tensorflow as tf

from time import strftime
from mysql.connector.pooling import MySQLConnectionPool

k_cols = ["lr"]

idxlst = None
feat_cols = []
parallel = None
time_shift = None
max_step = None
_prefetch = None
db_pool_size = None
db_host = None
db_port = None
db_pwd = None
shared_args = None
check_input = False

cnxpool = None

maxbno_query = ("SELECT  "
                "    vmax "
                "FROM "
                "    fs_stats "
                "WHERE "
                "    method = 'standardization' "
                "    AND tab = 'wcc_trn' "
                "    AND fields = %s ")


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
        query = ('SELECT distinct code COLLATE utf8mb4_0900_as_cs FROM idxlst')
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

def _init(db_pool_size=None, db_host=None, db_port=None, db_pwd=None):
    global cnxpool
    if cnxpool is None:
        print("{} [PID={}]: initializing mysql connection pool...".format(
            strftime("%H:%M:%S"), os.getpid()))
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
    ray.init(
        num_cpus=psutil.cpu_count(logical=False),
        webui_host='127.0.0.1',
        memory=4 * 1024 * 1024 * 1024,  # 4G
        object_store_memory=4 * 1024 * 1024 * 1024,  # 4G
        driver_object_store_memory=256 * 1024 * 1024    # 256M
    )


def _getDataSetMeta(flag):
    global cnxpool
    cnx = cnxpool.get_connection()
    max_bno, batch_size = None, None
    try:
        print('{} querying max batch no for {} set...'.format(
            strftime("%H:%M:%S"), flag))
        cursor = cnx.cursor()
        cursor.execute(maxbno_query, (flag + "_BNO", ))
        row = cursor.fetchone()
        max_bno = int(row[0])
        print('{} max batch no: {}'.format(strftime("%H:%M:%S"), max_bno))
        query = ("SELECT  "
                 "    COUNT(*) "
                 "FROM "
                 "    wcc_trn "
                 "WHERE "
                 "    flag = %s "
                 "    AND bno = 1 ")
        cursor.execute(query, (flag, ))
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

def getInputs(start_bno=0,
                 shift=0,
                 cols=None,
                 step=30,
                 cores=psutil.cpu_count(logical=False),
                 pfetch=2,
                 pool=None,
                 host=None,
                 port=None,
                 pwd=None,
                 vset=None,
                 check=False):
    """Input function for the stock trend prediction dataset.

    Returns:
        A dictionary of the following:
        'train': dataset for training
        'test': dataset for test/validation
        'train_batches': total batch of train set
        'test_batches': total batch of test set
        'train_batch_size': size of a single train set batch
        'test_batch_size': size of a single test set batch
    """
    # Create dataset for training
    global feat_cols, max_step, time_shift
    global parallel, _prefetch, db_pool_size
    global db_host, db_port, db_pwd, shared_args, check_input
    time_shift = shift
    feat_cols = cols or k_cols
    max_step = step
    feat_size = len(feat_cols) * 2 * (time_shift + 1)
    parallel = cores
    _prefetch = pfetch
    db_pool_size = pool
    db_host = host
    db_port = port
    db_pwd = pwd
    check_input = check
    print("{} Using parallel: {}, prefetch: {} db_host: {} port: {}".format(
        strftime("%H:%M:%S"), parallel, _prefetch, db_host, db_port))
    _init(db_pool_size, db_host, db_port, db_pwd)
    qk, qd, qd_idx, qk2 = _getFtQuery()
    shared_args = ray.put({
        'max_step': max_step,
        'time_shift': time_shift,
        'qk': qk,
        'qk2': qk2,
        'qd': qd,
        'qd_idx': qd_idx,
        'index_list': _getIndex(),
        'db_host': db_host,
        'db_port': db_port,
        'db_pwd': db_pwd
    })
    # query max flag from wcc_trn and fill a slice with flags between start and max
    train_batches, train_batch_size = _getDataSetMeta("TR")
    if train_batches is None:
        return None
    bnums = [bno for bno in range(start_bno, train_batches + 1)]

    def mapfunc(bno):
        ret = tf.numpy_function(func=_loadTrainingData_v2,
                                inp=[bno],
                                Tout=[tf.float32, tf.float32])
        feat, corl = ret
        feat.set_shape((None, max_step, feat_size))
        corl.set_shape((None, 1))
        return feat, corl

    ds_train = tf.data.Dataset.from_tensor_slices(bnums).map(
        lambda bno: tuple(mapfunc(bno)),
        # num_parallel_calls=tf.data.experimental.AUTOTUNE
        num_parallel_calls=parallel
    ).prefetch(
        # tf.data.experimental.AUTOTUNE
        _prefetch
    )

    # Create dataset for testing
    test_batches, test_batch_size = _getDataSetMeta("TS")
    ds_test = tf.data.Dataset.from_tensor_slices(
        _loadTestSet_v2(step, test_batches + 1,
                        vset)).batch(test_batch_size).cache().repeat()

    return {
        'train': ds_train,
        'test': ds_test,
        'train_batches': train_batches,
        'test_batches': test_batches,
        'train_batch_size': train_batch_size,
        'test_batch_size': test_batch_size
    }
