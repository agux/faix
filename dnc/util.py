# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DNC util ops and modules."""

# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def batch_invert_permutation(permutations):
    """Returns batched `tf.invert_permutation` for every row in `permutations`."""
    with tf.name_scope('batch_invert_permutation', values=[permutations]):
        # unpacked = tf.unstack(permutations)
        # inverses = [tf.invert_permutation(permutation) for permutation in unpacked]
        # return tf.stack(inverses)

        # fix performance issue
        perm = tf.cast(permutations, tf.float32)
        dim = int(perm.get_shape()[-1])
        size = tf.cast(tf.shape(perm)[0], tf.float32)
        delta = tf.cast(tf.shape(perm)[-1], tf.float32)
        rg = tf.range(0, size*delta, delta, dtype=tf.float32)
        rg = tf.reshape(rg, [-1, 1])
        rg = tf.tile(rg, [1, dim])
        perm = tf.add(perm, rg)
        flat = tf.reshape(perm, [-1])
        perm = tf.invert_permutation(tf.cast(flat, tf.int32))
        perm = tf.reshape(perm, [-1, dim])
        return tf.subtract(perm, tf.cast(rg, tf.int32))


def batch_gather(values, indices):
    """Returns batched `tf.gather` for every row in the input."""
    with tf.name_scope('batch_gather', values=[values, indices]):
        # unpacked = zip(tf.unstack(values), tf.unstack(indices))
        # result = [tf.gather(value, index) for value, index in unpacked]
        # return tf.stack(result)

        # fix performance issue
        idxf = tf.expand_dims(tf.cast(indices, tf.float32), -1)
        size = tf.shape(indices)[0]
        rg = tf.range(tf.cast(size, tf.float32), dtype=tf.float32)
        rg = tf.expand_dims(rg, -1)
        rg = tf.tile(rg, [1, int(indices.get_shape()[-1])])
        rg = tf.expand_dims(rg, -1)
        gidx = tf.cast(tf.concat([rg, idxf], -1), tf.int32)
        return tf.gather_nd(values, gidx)

def one_hot(length, index):
    """Return an nd array of given `length` filled with 0s and a 1 at `index`."""
    result=np.zeros(length)
    result[index]=1
    return result


def reduce_prod(x, axis, name=None):
    '''
    Uses tf.cumprod and tf.gather_nd as a workaround to calculating tf.reduce_prod's gradient
    on CPU.
    '''
    with tf.variable_scope(name or "c_reduce_prod"):
        cp=tf.cumprod(x, axis, reverse=True)
        size=tf.shape(cp)[0]
        idx1=tf.range(tf.cast(size, tf.float32), dtype=tf.float32)
        idx2=tf.zeros([size], tf.float32)
        # r=list(tf.map_fn(lambda p: (
        #     p[0], p[1]), (idx1, idx2), dtype=(tf.float32, tf.float32)))
        # indices=tf.stack(r, 1)
        indices = tf.stack([idx1, idx2], 1)
        return tf.gather_nd(cp, tf.cast(indices, tf.int32))
