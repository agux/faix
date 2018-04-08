from __future__ import print_function
import tensorflow as tf
from model import model, model2, model3, model4, model5, model6, model7, model8
from time import strftime
from data import data as data0
from data import data9, data10, data11
import os
import numpy as np
import math

EPOCH_SIZE = 444
LAYER_WIDTH = 128
RNN_LAYERS = 8
FCN_LAYERS = 16
MAX_STEP = 50
CARRY_BIAS = math.pi * math.e
# DROP_OUT = 0.1
LEARNING_RATE = 1e-3
LOG_DIR = 'logdir'


def collect_summary(sess, model, base_dir):
    train_writer = tf.summary.FileWriter(base_dir + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(base_dir + "/test", sess.graph)
    with tf.name_scope("Basic"):
        tf.summary.scalar("Loss", model.cost)
        tf.summary.scalar("Accuracy", model.accuracy*100)
    summary = tf.summary.merge_all()
    return summary, train_writer, test_writer


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    print('{} loading test data...'.format(strftime("%H:%M:%S")))
    tuuids, tdata, tlabels, tseqlen = data11.loadTestSet(MAX_STEP)
    print(tdata.shape)
    print(tlabels.shape)
    featSize = tdata.shape[2]
    nclass = tlabels.shape[1]
    classes = [i-nclass//2 for i in range(nclass)]
    data = tf.placeholder(tf.float32, [None, MAX_STEP, featSize], "input")
    target = tf.placeholder(tf.float32, [None, nclass], "labels")
    seqlen = tf.placeholder(tf.int32, [None], "seqlen")
    # dropout = tf.placeholder(tf.float32, name="dropout")
    # training = tf.placeholder(tf.bool, name="training")
    with tf.Session() as sess:
        model = model8.DRnnPredictorV3(
            data=data,
            target=target,
            seqlen=seqlen,
            classes=classes,
            layer_width=LAYER_WIDTH,
            num_rnn_layers=RNN_LAYERS,
            num_fcn_layers=FCN_LAYERS,
            carry_bias=CARRY_BIAS,
            learning_rate=LEARNING_RATE)
        stime = '{}'.format(strftime("%Y-%m-%d %H:%M:%S"))
        model_name = model.getName()
        base_dir = '{}/{}/{}'.format(LOG_DIR,
                                     model_name, strftime("%Y%m%d_%H%M%S"))
        print('{} using model: {}'.format(strftime("%H:%M:%S"), model_name))
        if tf.gfile.Exists(base_dir):
            tf.gfile.DeleteRecursively(base_dir)
        tf.gfile.MakeDirs(base_dir)
        # Isolate the variables stored behind the scenes by the metric operation
        metric_local_vars = tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES, scope="Precisions") + tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES, scope="Recalls")
        metric_vars_initializer = tf.variables_initializer(
            var_list=metric_local_vars)
        sess.run(tf.group(tf.global_variables_initializer(),
                          metric_vars_initializer))
        summary, train_writer, test_writer = collect_summary(
            sess, model, base_dir)
        saver = tf.train.Saver()
        bno = 0
        for epoch in range(EPOCH_SIZE):
            bno = epoch*50
            print('{} running on test set...'.format(strftime("%H:%M:%S")))
            feeds = {data: tdata, target: tlabels, seqlen: tseqlen}
            accuracy, worst, test_summary_str = sess.run(
                [model.accuracy, model.worst, summary, model.precisions[1], model.recalls[1], model.f_score], feeds)[:3]
            bidx, max_entropy, predict, actual = worst[0], worst[1], worst[2], worst[3]
            print('{} Epoch {} test accuracy {:3.3f}% max_entropy {:3.4f} predict {} actual {}'.format(
                strftime("%H:%M:%S"), epoch, 100. * accuracy, max_entropy, predict, actual))
            data0.save_worst_rec(model_name, stime, "test", epoch,
                                 tuuids[bidx], max_entropy, predict, actual)
            summary_str = None
            for i in range(50):
                sess.run(metric_vars_initializer)
                bno = bno+1
                print('{} loading training data for batch {}...'.format(
                    strftime("%H:%M:%S"), bno))
                truuids, trdata, labels, trseqlen = data11.loadTrainingData(
                    bno, MAX_STEP)
                print('{} training...'.format(strftime("%H:%M:%S")))
                feeds = {data: trdata, target: labels, seqlen: trseqlen}
                summary_str, worst = sess.run(
                    [summary, model.worst, model.optimize, model.precisions[1], model.recalls[1], model.f_score], feeds)[:2]
                bidx, max_entropy, predict, actual = worst[0], worst[1], worst[2], worst[3]
                print('{} bno {} max_entropy {:3.4f} predict {} actual {}'.format(
                    strftime("%H:%M:%S"), bno, max_entropy, predict, actual))
                data0.save_worst_rec(model_name, stime, "train", bno,
                                     truuids[bidx], max_entropy, predict, actual)
                train_writer.add_summary(summary_str, bno)
                test_writer.add_summary(test_summary_str, bno)
                train_writer.flush()
                test_writer.flush()
            checkpoint_file = os.path.join(base_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=bno)
            sess.run(metric_vars_initializer)
        # test last epoch
        print('{} running on test set...'.format(strftime("%H:%M:%S")))
        feeds = {data: tdata, target: tlabels, seqlen: tseqlen}
        accuracy, worst, test_summary_str = sess.run(
            [model.accuracy, model.worst, summary, model.precisions[1], model.recalls[1], model.f_score], feeds)[:3]
        bidx, max_entropy, predict, actual = worst[0], worst[1], worst[2], worst[3]
        print('{} Epoch {} test accuracy {:3.3f}% max_entropy {:3.4f} predict {} actual {}'.format(
            strftime("%H:%M:%S"), EPOCH_SIZE, 100. * accuracy, max_entropy, predict, actual))
        data0.save_worst_rec(model_name, stime, "test", EPOCH_SIZE,
                             tuuids[bidx], max_entropy, predict, actual)
        train_writer.add_summary(summary_str, bno)
        test_writer.add_summary(test_summary_str, bno)
        train_writer.flush()
        test_writer.flush()
