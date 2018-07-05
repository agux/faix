"""Create the input data pipeline using `tf.data`"""

from time import strftime
# from joblib import Parallel, delayed
import tensorflow as tf
import sys
import os
import multiprocessing
import numpy as np
import glob
import re
import gzip
import json
import ConfigParser

file_dir = None


def _loadTestSet(vset=None):
    global file_dir
    setno = vset
    if setno is None:
        files = glob.glob(os.path.join(file_dir, 'TEST_*.json.gz'))
        ntest = max([int(re.search('TEST_(.+?).json.gz', path).group(1))
                     for path in files])
        setno = np.random.randint(ntest)
    flag = 'TEST_{}'.format(setno)
    print('{} selected test set: {}'.format(
        strftime("%H:%M:%S"), flag))
    # load json data from gz file
    file = os.path.join(file_dir, '{}.json.gz'.format(flag))
    with gzip.GzipFile(file, 'rb') as fin:
        data = json.loads(fin.read().decode('utf-8'))
    # data = [batch, max_step, feature*time_shift]
    # vals = [batch]
    # seqlen = [batch]
    return np.array(data['features'], 'f'), np.array(data['labels'], 'f'), np.array(data['seqlens'], 'i')


def _loadTrainingData(flag):
    global file_dir
    print("{} loading training set {}...".format(
        strftime("%H:%M:%S"), flag))
    file = os.path.join(file_dir, '{}.json.gz'.format(flag))
    with gzip.GzipFile(file, 'rb') as fin:
        data = json.loads(fin.read().decode('utf-8'))
    # data = [batch, max_step, feature*time_shift]
    # vals = [batch]
    # seqlen = [batch]
    return np.array(data['features'], 'f'), np.array(data['labels'], 'f'), np.array(data['seqlens'], 'i')


def getInputs(dir, start=0, prefetch=2, vset=None):
    """Input function for the wcc training dataset.

    Returns:
        A dictionary containing:
        features,labels,seqlens,handle,train_iter,test_iter
    """
    global file_dir
    file_dir = dir
    # read meta.txt from file_dir
    config = ConfigParser.ConfigParser()
    config.readfp(open(os.path.join(file_dir, 'meta.txt'), 'r'))
    time_step = config.getint('common', 'time_step')
    feature_size = config.getint('common', 'feature_size')
    max_bno = config.getint('training set', 'count')
    test_batch_size = config.getint('test set', 'batch_size')
    # Create dataset for training
    print("{} Using prefetch: {}".format(
        strftime("%H:%M:%S"), prefetch))
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
            _loadTestSet(vset)).batch(test_batch_size).repeat()

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
