from __future__ import print_function
import __main__ as main
import pandas as pd
import mysql.connector
import numpy as np
import tensorflow as tf
from pstk import data as dat
from sqlalchemy import create_engine
from joblib import Parallel, delayed


def testGetFileName():
    print(main.__file__)


def testEmbedding():
    sess = tf.InteractiveSession()
    word_embeddings = tf.get_variable(
        "word_embeddings", [5, 5], initializer=tf.truncated_normal_initializer())
    sess.run(tf.global_variables_initializer())
    word_ids = [0, 0, 1, 2]
    embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)
    r = embedded_word_ids.eval()
    print("{}".format(r))
    print(r.shape)


def getIndices(x, n):
    print("x:{}".format(x.get_shape()))
    indices = tf.stack([tf.fill([n], x[0]), [
        x[1]-n+i for i in range(n)]], axis=1)
    print(indices.get_shape())
    return indices


def testGatherND():
    # gather last 2 elements of (2, 5, 2), -> (2, 2, 2)
    # indices = np.asarray([[[0, 1], [0, 2]], [[1, 0], [1, 1]]])
    params = np.asarray([[['a0', 'b0'], ['c0', 'd0'], ['e0', 'f0'], ['g0', 'h0'], ['i0', 'j0']],
                         [['a1', 'b1'], ['c1', 'd1'], ['e1', 'f1'], ['g1', 'h1'], ['0', '0']]])
    batch = 2
    n = 2
    length = tf.placeholder(tf.int32, shape=[None])
    mapinput = tf.stack([tf.range(batch), length], axis=1)
    print("mapinput: {}".format(mapinput.get_shape()))
    indices = tf.map_fn(lambda x: getIndices(
        x, n), mapinput)
    # [tf.stack([tf.constant(b, shape=[batch]), [
    #                 s-n+i for i in range(n)]], axis=1) for b, s in enumerate(length)]
    sess = tf.InteractiveSession()
    gnd = tf.gather_nd(params, indices)
    i, r = sess.run([indices, gnd], feed_dict={length: [5, 4]})
    print(i)
    print(params.shape)
    print(r.shape)
    print("{}".format(r))


def testTensorShape():
    x = tf.placeholder(shape=[None, 16], dtype=tf.float32)
    d = tf.placeholder(shape=[], dtype=tf.float32)
    random_tensor = tf.random_uniform(tf.shape(x), dtype=tf.float32)
    print("random_tensor: {}".format(random_tensor.get_shape()))
    kept_idx = tf.greater_equal(random_tensor, 1.0 - d)
    print("kept_idx: {}".format(kept_idx.get_shape()))


def delayedFunc(i):
    return i+1, i*2, i**2


def testJoblib():
    r = Parallel(n_jobs=5)(delayed(delayedFunc)(i) for i in range(30))
    r1, r2, r3 = zip(*r)
    print("r1 ({}):{}".format(type(list(r1)), r1))
    print("r2:{}".format(r2))
    print("r3:{}".format(r3))


def testVariableScope():
    a = 3
    if 1 < 3:
        a = a+1
    else:
        a = a-1
    print(a)

# testGatherND()
# testGetFileName()
# print(__file__)
# f = __file__
# print(f[f.rindex('/')+1:f.rindex('.py')])

# testTensorShape()


# testJoblib()

testVariableScope()
