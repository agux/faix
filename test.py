from __future__ import print_function
import pandas as pd
import mysql.connector
import numpy as np
import tensorflow as tf
from pstk import data as dat
from sqlalchemy import create_engine


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


testGatherND()
