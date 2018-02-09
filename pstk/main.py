from __future__ import print_function
import tensorflow as tf
from model import SecurityGradePredictor
from time import strftime
import data as dat
import os

EPOCH_SIZE = 4130
HIDDEN_SIZE = 200
NUM_LAYERS = 2
#BATCH_SIZE = 200
MAX_STEP = 500
FEAT_SIZE = 20
NUM_CLASSES = 21
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-3
LOG_DIR = 'logdir'

if __name__ == '__main__':
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)

    data = tf.placeholder(tf.float32, [None, MAX_STEP, FEAT_SIZE])
    target = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    dropout = tf.placeholder(tf.float32)
    model = SecurityGradePredictor(
        data, target, dropout, num_hidden=HIDDEN_SIZE, num_layers=NUM_LAYERS, learning_rate=LEARNING_RATE)
    tf.summary.scalar("Training Loss", model.cost)
    tf.summary.scalar("Train Error", model.error*100)
    summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    print('{} loading test data...'.format(strftime("%H:%M:%S")))
    _, tdata, tlabels = dat.loadTestSet(MAX_STEP)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())
        bno = 0
        for epoch in range(EPOCH_SIZE):
            bno = epoch*10
            for i in range(10):
                bno = bno+1
                print('{} loading training data for batch {}...'.format(
                    strftime("%H:%M:%S"), bno))
                uuid, trdata, labels = dat.loadPrepTrainingData(bno, MAX_STEP)
                print('{} training...'.format(strftime("%H:%M:%S")))
                summary_str, _ = sess.run([summary, model.optimize], {
                                        data: trdata, target: labels, dropout: DROPOUT_RATE})
                summary_writer.add_summary(summary_str, bno)
                summary_writer.flush()
                # print('{} tagging data as trained, batch no: {}'.format(
                #     strftime("%H:%M:%S"), bno))
                # dat.tagDataTrained(uuid, bno)
            print('{} running on test set...'.format(strftime("%H:%M:%S")))
            error = sess.run(
                model.error, {data: tdata, target: tlabels, dropout: 0})
            print('{} Epoch {:2d} error {:3.1f}%'.format(
                strftime("%H:%M:%S"), epoch + 1, 100 * error))
            checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=bno)
