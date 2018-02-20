from __future__ import print_function
import tensorflow as tf
from model import BasicSecGradePredictor
from time import strftime
import os
import numpy as np


print("Expected cross entropy loss if the model:")
print("- learns neither dependency:", -(0.625 * np.log(0.625) +
                                        0.375 * np.log(0.375)))
# Learns first dependency only ==> 0.51916669970720941
print("- learns first dependency:  ",
      -0.5 * (0.875 * np.log(0.875) + 0.125 * np.log(0.125))
      - 0.5 * (0.625 * np.log(0.625) + 0.375 * np.log(0.375)))
print("- learns both dependencies: ", -0.50 * (0.75 * np.log(0.75) + 0.25 * np.log(0.25))
      - 0.25 * (2 * 0.50 * np.log(0.50)) - 0.25 * (0))


EPOCH_SIZE = 4130
HIDDEN_SIZE = 256
NUM_LAYERS = 4
BATCH_SIZE = 200
W_SIZE = 10
MAX_STEP = 200
num_classes = 2
LEARNING_RATE = 1e-3
LOG_DIR = 'logdir'


def gen_data(size=50000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i - 3] == 1:
            threshold += 0.5
        if X[i - 8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length *
                          i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length *
                          i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = np.zeros([batch_size, num_steps, num_classes], dtype=np.int32)
        y = np.zeros([batch_size, num_classes], dtype=np.int32)
        for b in range(batch_size):
            step = np.random.randint(num_steps//2, num_steps)
            fs = 0
            for s in range(step):
                x[b][s][data_x[b][i*num_steps+s]] = 1
                fs = s
            y[b][data_y[b][i*num_steps+fs-1]] = 1
        yield (x, y)


def gen_epochs(n, num_steps):
    for _ in range(n):
        yield gen_batch(gen_data(), BATCH_SIZE, num_steps)


if __name__ == '__main__':
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    data = tf.placeholder(tf.float32, [None, MAX_STEP, num_classes])
    target = tf.placeholder(tf.float32, [None, num_classes])
    training = tf.placeholder(tf.bool)
    model = BasicSecGradePredictor(
        data, target, W_SIZE, training, num_hidden=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, learning_rate=LEARNING_RATE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        tf.summary.scalar("Train Loss", model.cost)
        tf.summary.scalar("Train Accuracy", model.accuracy*100)
        summary = tf.summary.merge_all()
        saver = tf.train.Saver()
        for idx, epoch in enumerate(gen_epochs(1, MAX_STEP)):
            print("\nEPOCH", idx)
            for step, (X, Y) in enumerate(epoch):
                summary_str, _ = sess.run([summary, model.optimize],
                                          feed_dict={data: X, target: Y, training: True})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
