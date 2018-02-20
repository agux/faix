from __future__ import print_function
import pandas as pd
import mysql.connector
import numpy as np
import tensorflow as tf
from pstk import data as dat
from sqlalchemy import create_engine

sess = tf.InteractiveSession()
word_embeddings = tf.get_variable(
    "word_embeddings", [5, 5], initializer=tf.truncated_normal_initializer())
sess.run(tf.global_variables_initializer())
word_ids = [0,0,1,2]
embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)
r = embedded_word_ids.eval()
print("{}".format(r))
print(r.shape)
