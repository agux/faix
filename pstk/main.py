from __future__ import print_function
import tensorflow as tf
from model import SecurityGradePredictor
from time import strftime
import data as dat

HIDDEN_SIZE = 500
BATCH_SIZE = 100
MAX_STEP = 1000
FEAT_SIZE = 11
NUM_CLASSES = 21

if __name__ == '__main__':
    data = tf.placeholder(tf.float32, [None, MAX_STEP, FEAT_SIZE])
    target = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    model = SecurityGradePredictor(data, target,num_hidden=HIDDEN_SIZE)
    print('{} loading test data...'.format(strftime("%H:%M:%S")))
    _, tdata, tlabels = dat.loadTestSet(BATCH_SIZE*10, MAX_STEP)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(7000):
        for i in range(10):
            bno = epoch*10+i+1
            print('{} loading training data for batch {}...'.format(strftime("%H:%M:%S"), bno))
            uuid, trdata, labels = dat.loadTrainingData(BATCH_SIZE, MAX_STEP)
            print('{} training...'.format(strftime("%H:%M:%S")))
            sess.run(model.optimize, {data: trdata, target: labels})
            print('{} tagging data as trained, batch no: {}'.format(strftime("%H:%M:%S"), bno))
            dat.tagDataTrained(uuid, bno)
        error = sess.run(model.error, {data: tdata, target: tlabels})
        print('{} Epoch {:2d} error {:3.1f}%'.format(strftime("%H:%M:%S"), epoch + 1, 100 * error))