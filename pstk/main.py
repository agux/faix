from __future__ import print_function
import tensorflow as tf
from model import SecurityGradePredictor
from time import strftime
import data as dat
import os

EPOCH_SIZE = 4130
HIDDEN_SIZE = 512
NUM_LAYERS = 3
#BATCH_SIZE = 200
W_SIZE = 10
MAX_STEP = 60
FEAT_SIZE = 30
NUM_CLASSES = 21
LEARNING_RATE = 1e-4
LOG_DIR = 'logdir'

if __name__ == '__main__':
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)

    data = tf.placeholder(tf.float32, [None, MAX_STEP, FEAT_SIZE])
    target = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    training = tf.placeholder(tf.bool)
    model = SecurityGradePredictor(
        data, target, W_SIZE, training, True, num_hidden=HIDDEN_SIZE, num_layers=NUM_LAYERS, learning_rate=LEARNING_RATE)
    tf.summary.scalar("Train Loss", model.cost)
    tf.summary.scalar("Train Accuracy", model.accuracy*100)
    summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    print('{} loading test data...'.format(strftime("%H:%M:%S")))
    _, tdata, tlabels = dat.loadTestSet(MAX_STEP)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())
        bno = 0
        for epoch in range(EPOCH_SIZE):
            bno = epoch*50
            for i in range(50):
                bno = bno+1
                print('{} loading training data for batch {}...'.format(
                    strftime("%H:%M:%S"), bno))
                uuid, trdata, labels = dat.loadPrepTrainingData(bno, MAX_STEP)
                print('{} training...'.format(strftime("%H:%M:%S")))
                summary_str, _ = sess.run([summary, model.optimize], {
                    data: trdata, target: labels, training: True})
                summary_writer.add_summary(summary_str, bno)
                summary_writer.flush()
                # print('{} tagging data as trained, batch no: {}'.format(
                #     strftime("%H:%M:%S"), bno))
                # dat.tagDataTrained(uuid, bno)
            print('{} running on test set...'.format(strftime("%H:%M:%S")))
            accuracy = sess.run(
                model.accuracy, {data: tdata, target: tlabels, training: False})
            print('{} Epoch {:4d} test accuracy {:3.3f}%'.format(
                strftime("%H:%M:%S"), epoch + 1, 100 * accuracy))
            checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=bno)
