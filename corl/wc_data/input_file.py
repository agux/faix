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
    global file_dir
    setno = vset
    if setno is None:
        setno = np.random.randint(max_bno)
    flag = 'TEST_{}'.format(setno)
    print('{} selected test set: {}'.format(
        strftime("%H:%M:%S"), flag))
    # load json data from local or remote gz file
    data = _loadData(file_dir, flag)
    # data = [batch, max_step, feature*time_shift]
    # vals = [batch]
    # seqlen = [batch]
    return np.array(data['features'], 'f'), np.array(data['labels'], 'f'), np.array(data['seqlens'], 'i')


def _loadTrainingData(flag):
    global file_dir
    print("{} loading training set {}...".format(
        strftime("%H:%M:%S"), flag))
    # load json data from local or remote gz file
    data = _loadData(file_dir, flag)
    # data = [batch, max_step, feature*time_shift]
    # vals = [batch]
    # seqlen = [batch]
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
    max_bno = config.getint('training set', 'count')
    test_batch_size = config.getint('test set', 'batch_size')
    test_max_bno = config.getint('test set', 'count')
    return time_step, feature_size, max_bno, test_batch_size, test_max_bno


def getInputs(dir, start=0, prefetch=2, vset=None):
    """Input function for the wcc training dataset.

    Returns:
        A dictionary containing:
        features,labels,seqlens,handle,train_iter,test_iter
    """
    global file_dir
    file_dir = dir
    print("{} loading file from: {} Start from: {} Using prefetch: {}".format(
        strftime("%H:%M:%S"), file_dir, start, prefetch))
    # read meta.txt from file_dir
    time_step, feature_size, max_bno, test_batch_size, test_max_bno = _read_meta_config(
        file_dir)
    # Create dataset for training
    with tf.variable_scope("build_inputs"):
        # query max flag from wcc_trn and fill a slice with flags between start and max
        flags = ["TRAIN_{}".format(bno) for bno in range(start, max_bno)]
        train_dataset = tf.data.Dataset.from_tensor_slices(flags).map(
            lambda f: tuple(
                tf.py_func(_loadTrainingData, [f], [
                           tf.float32, tf.float32, tf.int32])
            ), prefetch
        ).batch(1).prefetch(prefetch)
        # Create dataset for testing
        test_dataset = tf.data.Dataset.from_tensor_slices(
            _loadTestSet(test_max_bno, vset)).batch(test_batch_size).repeat()

        train_iterator = train_dataset.make_one_shot_iterator()
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
                  'train_iter': train_iterator, 'test_iter': test_iterator}
        return inputs
