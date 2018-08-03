"""Create the input data pipeline using `tf.data`"""

from time import strftime
from google.cloud import storage as gcs
from retrying import retry
import tensorflow as tf
import sys
import os
import re
import multiprocessing
import numpy as np
import gzip
import json
import ConfigParser
import tempfile as tmpf

file_dir = None
vol_size = None
gcs_client = None


def print_n_retry(exception):
    print(exception)
    return True


@retry(retry_on_exception=print_n_retry,
       stop_max_attempt_number=7,
       wait_exponential_multiplier=1000,
       wait_exponential_max=32000)
def _file_from_gcs(bucket_name, object_name, spooled=True):
    global gcs_client
    if gcs_client is None:
        gcs_client = gcs.Client()
    bucket = gcs_client.get_bucket(bucket_name)
    blob = bucket.blob(object_name)
    tmp = tmpf.SpooledTemporaryFile(
        max_size=1024*1024*100) if spooled else tmpf.NamedTemporaryFile()
    blob.download_to_file(tmp)
    tmp.seek(0)
    return tmp


def _loadData(file_dir, flag):
    '''
    Load json data from local or remote gz file

    Returned dict structure:
    features = [batch, max_step, feature*time_shift]
    labels = [batch]
    seqlens = [batch]
    '''
    file = None
    if file_dir.startswith('gs://'):
        s = re.search('gs://([^/]*)/(.*)', file_dir)
        bn = s.group(1)
        on = '{}/{}.json.gz'.format(s.group(2), flag)
        with gzip.GzipFile(fileobj=_file_from_gcs(bn, on), mode='rb') as fin:
            return json.loads(fin.read().decode('utf-8'))
    else:
        file = os.path.join(file_dir, '{}.json.gz'.format(flag))
        with gzip.GzipFile(file, 'rb') as fin:
            return json.loads(fin.read().decode('utf-8'))


def _loadTestSet(max_bno, vset=None):
    global file_dir, vol_size
    if vset is None:
        vset = np.random.randint(max_bno)
    flag = 'TEST_{}'.format(vset)
    print('{} selected test set: {}'.format(
        strftime("%H:%M:%S"), flag))
    path = file_dir
    if vol_size is not None:
        path = "{}/vol_{}".format(file_dir, vset//vol_size)
    data = _loadData(path, flag)
    return np.array(data['features'], 'f'), np.array(data['labels'], 'f'), np.array(data['seqlens'], 'i')


def _loadBatchData(flag):
    global file_dir, vol_size
    print("{} loading dataset {}...".format(
        strftime("%H:%M:%S"), flag))
    path = file_dir
    if vol_size is not None:
        bno = int(flag[flag.rfind("_")+1:])
        path = "{}/vol_{}".format(file_dir, bno//vol_size)
    data = _loadData(path, flag)
    return np.array(data['features'], 'f'), np.array(data['labels'], 'f'), np.array(data['seqlens'], 'i')


def _read_meta_config(file_dir):
    file = None
    config = ConfigParser.RawConfigParser()
    if file_dir.startswith('gs://'):
        s = re.search('gs://([^/]*)/(.*)', file_dir)
        bn = s.group(1)
        on = '{}/meta.txt'.format(s.group(2))
        file = _file_from_gcs(bn, on)
    else:
        file = open(os.path.join(file_dir, 'meta.txt'), 'rb')
    config.readfp(file)
    file.seek(0)
    print('{} read meta.txt:\n{}'.format(strftime("%H:%M:%S"), file.read()))
    file.close()
    time_step = config.getint('common', 'time_step')
    feature_size = config.getint('common', 'feature_size')
    train_batch_size = config.getint('training set', 'batch_size')
    train_max_bno = config.getint('training set', 'count')
    test_batch_size = config.getint('test set', 'batch_size')
    test_max_bno = config.getint('test set', 'count')
    infer_flags = config.get('infer set', 'flags').split(
    ) if config.has_option('infer set', 'flags') else []
    return time_step, feature_size, train_batch_size, train_max_bno, test_batch_size, test_max_bno, infer_flags


def _map_func(flag):
    return tuple(tf.py_func(_loadBatchData, [flag], [tf.float32, tf.float32, tf.int32]))


def getInputs(path, start=0, prefetch=2, vset=None, volsize=None):
    """Input function for the wcc training dataset.

    Returns:
        A dictionary containing:
        features,labels,seqlens,handle,train_iter,test_iter,infer_iter
    """
    global file_dir, vol_size
    file_dir = path
    vol_size = volsize
    print("{} loading file from: {} Start from: {} Using prefetch: {}".format(
        strftime("%H:%M:%S"), file_dir, start, prefetch))
    # read meta.txt from file_dir
    mc = _read_meta_config(file_dir)
    time_step = mc[0]
    feature_size = mc[1]
    train_batch_size = mc[2]
    max_bno = mc[3]
    test_batch_size = mc[4]
    test_max_bno = mc[5]
    infer_flags = mc[6]
    cpu = multiprocessing.cpu_count()
    with tf.variable_scope("build_inputs"):
        # Create dataset for training
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ["TRAIN_{}".format(bno) for bno in range(start, max_bno)]
        ).map(_map_func, cpu).batch(1).prefetch(prefetch)
        # Create dataset for inference
        infer_dataset = tf.data.Dataset.from_tensor_slices(
            infer_flags
        ).map(_map_func, cpu).batch(1).prefetch(prefetch)
        # Create dataset for testing
        test_dataset = tf.data.Dataset.from_tensor_slices(
            _loadTestSet(test_max_bno, vset)
        ).batch(test_batch_size).repeat()

        train_iterator = train_dataset.make_one_shot_iterator()
        infer_iterator = infer_dataset.make_one_shot_iterator()
        test_iterator = test_dataset.make_one_shot_iterator()

        handle = tf.placeholder(tf.string, shape=[])
        types = (tf.float32, tf.float32, tf.int32)
        shapes = (tf.TensorShape([None, time_step, feature_size]),
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
                  'train_iter': train_iterator, 'test_iter': test_iterator, 'infer_iter': infer_iterator,
                  'train_batch_size': train_batch_size, 'test_batch_size': test_batch_size}
        return inputs
