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
header_size = 0
line_size = 0

TASKLIST_FILE = 'wccinfer_tasklist'
TALIST_SEP = ' | '
gs_infer_base_path = None


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


@retry(retry_on_exception=print_n_retry,
       stop_max_attempt_number=7,
       wait_exponential_multiplier=1000,
       wait_exponential_max=32000)
def _scan_gcs(bucket_name, prefix, project=None):
    global gcs_client
    if gcs_client is None:
        gcs_client = gcs.Client(project)
    bucket = gcs_client.get_bucket(bucket_name)
    return bucket.list_blobs(prefix=prefix)


def _get_infer_tasklist(rbase, project=None):
    '''
    Returns string array of [file_id, index]
    '''
    global header_size, line_size, gs_infer_base_path
    TALIST_SEP = ' | '
    tasklist = []
    if os.path.exists(TASKLIST_FILE):
        print('{} tasklist found, parsing...'.format(strftime("%H:%M:%S")))
        with open(TASKLIST_FILE, 'rb') as f:
            h = f.readline()[:-1]  # strip line break
            header_size = len(h)
            fs = [field.strip() for field in h.split(TALIST_SEP)]
            line_size = int(fs[2])
            print('{} base path: {} total: {} header size: {} line size: {}'.format(
                strftime("%H:%M:%S"), fs[0], fs[1], header_size, line_size))
            gs_infer_base_path = fs[0]
            print('{} scanning tasklist file...'.format(strftime("%H:%M:%S")))
            lc = 0
            tasklist = []
            for line in f:
                fs = [field.strip() for field in line.split(TALIST_SEP)]
                if fs[1] == "P":
                    tasklist.append(
                        [fs[0], str(header_size+1 + lc*(line_size+1))])
                lc = lc + 1
            print('{} pending task: {}'.format(
                strftime("%H:%M:%S"), len(tasklist)))
    else:
        print('{} tasklist not present, scanning files in {}...'.format(
            strftime("%H:%M:%S"), rbase))
        gs_infer_base_path = rbase
        s = re.search('gs://([^/]*)/(.*)', rbase)
        bn = s.group(1)
        folder = s.group(2)
        prefix = '{}/vol_'.format(folder)
        blobs = _scan_gcs(bn, prefix, project)
        print('{} constructing tasklist...'.format(strftime("%H:%M:%S")))
        idx = len('{}/{}'.format(bn, prefix))
        relpaths = []
        line_size = 0
        for b in blobs:
            # strip bucket name and generation number
            rp = b.id[idx:b.id.rfind('.json.gz/')]
            relpaths.append(rp)
            line_size = max(line_size, len(rp+TALIST_SEP+'P'))
        total = len(relpaths)
        header = '{}{}{}{}{}'.format(
            rbase, TALIST_SEP, total, TALIST_SEP, line_size)
        header_size = len(header)
        print('{} base path: {} total: {} header size: {} line size: {}'.format(
            strftime("%H:%M:%S"), rbase, total, header_size, line_size))
        tasklist = []
        with open(TASKLIST_FILE, 'wb') as f:
            f.write(header + '\n')
            for i, p in enumerate(relpaths):
                f.write('{}{}P'.format(p, TALIST_SEP).ljust(line_size) + '\n')
                tasklist.append([p, str(header_size+1 + i*(line_size+1))])
            f.flush()
        print('{} tasklist file generated'.format(strftime("%H:%M:%S")))

    return np.array(tasklist, 'U')


def _loadData(file_dir, file_name):
    '''
    Load json data from local or remote gz file

    Returned dict structure:
    features = [None, max_step, feature*time_shift]
    labels = [None] # for training and test set
    seqlens = [None]
    code = [] # for inference set
    klid = [] # for inference set
    refs = [None] # for inference set
    '''
    file = None
    if file_dir.startswith('gs://'):
        s = re.search('gs://([^/]*)/(.*)', file_dir)
        bn = s.group(1)
        on = '{}/{}.json.gz'.format(s.group(2), file_name)
        with gzip.GzipFile(fileobj=_file_from_gcs(bn, on), mode='rb') as fin:
            return json.loads(fin.read().decode('utf-8'))
    else:
        file = os.path.join(file_dir, '{}.json.gz'.format(file_name))
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


def _load_infer_data(task):
    global gs_infer_base_path
    v, bname = os.path.split(task[0])
    path = '{}/vol_{}'.format(gs_infer_base_path, v)
    name = '{}.json.gz'.format(bname)
    print("{} loading infer file {}/{}...".format(strftime("%H:%M:%S"), path, name))
    data = _loadData(path, name)
    # features, seqlens, code, klid, refs, idx
    return np.array(data['features'], 'f'), np.array(data['seqlens'], 'i'), data['code'], \
        data['klid'], np.array(data['refs'], 'U'), int(task[1])


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


def _infer_map_func(task):
    # features, seqlens, code, klid, refs, idx
    return tuple(tf.py_func(_load_infer_data, [task], [tf.float32, tf.int32, tf.string, tf.int32, tf.string, tf.int32]))


def _map_func(flag):
    return tuple(tf.py_func(_loadBatchData, [flag], [tf.float32, tf.float32, tf.int32]))


def getInferInput(rbase, prefetch=2):
    """Input function for the wcc inference dataset.

    Returns:
        A dictionary containing:
        features,seqlens,handle,infer_iter,code,klid,idx
    """
    print("{} loading file from: {} Using prefetch: {}".format(
        strftime("%H:%M:%S"), rbase, prefetch))
    mc = _read_meta_config(rbase)
    time_step = mc[0]
    feature_size = mc[1]
    cpu = multiprocessing.cpu_count()
    with tf.variable_scope("build_inputs"):
        # Create dataset for inference
        infer_dataset = tf.data.Dataset.from_tensor_slices(
            _get_infer_tasklist(rbase)
        ).map(_infer_map_func, cpu).batch(1).prefetch(prefetch)

        infer_iterator = infer_dataset.make_one_shot_iterator()

        handle = tf.placeholder(tf.string, shape=[])
        # features, seqlens, code, klid, refs, idx
        types = (tf.float32, tf.int32, tf.string,
                 tf.int32, tf.string, tf.int32)
        shapes = (tf.TensorShape([None, time_step, feature_size]),
                  tf.TensorShape([None]),
                  tf.TensorShape([]),
                  tf.TensorShape([]),
                  tf.TensorShape([None]),
                  tf.TensorShape([]))
        iter = tf.data.Iterator.from_string_handle(
            handle, types, infer_iterator.output_shapes)

        next_el = iter.get_next()
        features = tf.squeeze(next_el[0])
        features.set_shape(shapes[0])
        seqlens = tf.squeeze(next_el[1])
        seqlens.set_shape(shapes[1])
        code = tf.squeeze(next_el[2])
        code.set_shape(shapes[2])
        klid = tf.squeeze(next_el[3])
        klid.set_shape(shapes[3])
        refs = tf.squeeze(next_el[4])
        refs.set_shape(shapes[4])
        idx = tf.squeeze(next_el[5])
        idx.set_shape(shapes[5])
        # return a dictionary
        inputs = {'features': features, 'seqlens': seqlens, 'handle': handle,
                  'infer_iter': infer_iterator, 'code': code, 'klid': klid, 'refs': refs, 'idx': idx}
        return inputs


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


def check_task_status():
    for i in range(5):
        ptask = []
        print("{} #{} scanning unfinished tasks...".format(
            strftime("%H:%M:%S"), i))
        with open(TASKLIST_FILE, 'rb') as f:
            ln = 0
            for ln in f:
                ln = ln + 1
                # skip header line
                if ln == 1:
                    continue
                fs = [field.strip() for field in ln.split(TALIST_SEP)]
                if fs[1] == "P":
                    ptask.append({'path': fs[0], 'ln': ln})
        if len(ptask) > 0:
            print("{} #{} scan results: {} tasks pending".format(
                strftime("%H:%M:%S"), i, len(ptask)))
        else:
            # no more pending tasks
            print("{} all tasks have completed".format(strftime("%H:%M:%S")))
            return
    # if ptask is not empty, print them individually
    if len(ptask) > 0:
        for t in ptask:
            print(t)
    else:
        # no more pending tasks
        print("{} all tasks have completed".format(strftime("%H:%M:%S")))
