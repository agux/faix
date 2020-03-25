from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
import random
import shutil
import math
from common import parseArgs, LOG_DIR
from wc_data import input_fn
from time import strftime
from model.tf2 import lstm
import tensorflow as tf
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# pylint: disable-msg=E0401

VSET = 5
TEST_BATCH_SIZE = 3000
TEST_INTERVAL = 50
TRACE_INTERVAL = 10
SAVE_INTERVAL = 20
LAYER_WIDTH = 512
MEMORY_SIZE = 32
WORD_SIZE = 32
NUM_WRITES = 2
NUM_READS = 8
MAX_STEP = 35
TIME_SHIFT = 4
KEEP_PROB = 0.5
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-3
LR_DECAY_STEPS = 1000
DECAYED_LR_START = 40000
DROPOUT_DECAY_STEPS = 1000
DECAYED_DROPOUT_START = 40000
SEED = 285139

VAL_SAVE_FREQ = 50

feat_cols = ["close", "volume", "amount"]

# pylint: disable-msg=E0601,E1101

def getInput(start, args):
    ds = args.ds.lower()
    print('{} using data source: {}'.format(strftime("%H:%M:%S"), args.ds))
    if ds == 'db':
        return input_fn.getInputs(TIME_SHIFT, feat_cols, MAX_STEP,
                                  args.parallel, args.prefetch, args.db_pool,
                                  args.db_host, args.db_port, args.db_pwd,
                                  args.vset or VSET)
    return None

def run(args):
    print("{} started training, pid:{}".format(strftime("%H:%M:%S"),
                                               os.getpid()))
    regressor = lstm.LSTMRegressorV1(
        layer_width=LAYER_WIDTH,
        dropout_rate=DROPOUT_RATE,
        decayed_dropout_start=DECAYED_DROPOUT_START,
        dropout_decay_steps=DROPOUT_DECAY_STEPS,
        learning_rate=LEARNING_RATE,
        decayed_lr_start=DECAYED_LR_START,
        lr_decay_steps=LR_DECAY_STEPS,
        seed=SEED)
    model_name = regressor.getName()
    print('{} using model: {}'.format(strftime("%H:%M:%S"), model_name))

    f = __file__
    testn = f[f.rfind('/') + 1:f.rindex('.py')]
    base_name = "{}_{}".format(testn, model_name)
    base_dir = os.path.join(LOG_DIR, base_name)
    # Define the checkpoint directory to store the checkpoints
    log_dir = os.path.join(base_dir, 'tblogs')
    training_dir = os.path.join(base_dir, 'training')
    best_dir = os.path.join(base_dir, 'best')
    # Function for decaying the learning rate.
    # You can define any decay function you need.
    # decay = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
    #     min_delta=0.0001, cooldown=0, min_lr=0, **kwargs
    # )
    tensorboard_cbk = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        # how often to log histogram visualizations
        histogram_freq=VAL_SAVE_FREQ,
        #the log file can become quite large when write_graph is set to True.
        write_graph=True,
        #whether to write model weights to visualize as image in TensorBoard.
        write_images=False,
        # How often to write logs (default: once per epoch)
        update_freq='epoch')
    callbacks = [
        # decay,
        tensorboard_cbk,
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
            training_dir, "ckpt_{epoch}_{val_loss:.5f}.tf"),
                                           monitor='val_loss',
                                           verbose=1,
                                           save_freq=VAL_SAVE_FREQ),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
            best_dir, "ckpt_best.tf"),
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_freq=VAL_SAVE_FREQ)
    ]
    # Check if previous training progress exists
    bno = 0
    restored = False
    latest_ckpt = tf.train.get_checkpoint_state(training_dir)
    if tf.io.gfile.exists(training_dir):
        print("{} training folder exists".format(strftime("%H:%M:%S")))
        if latest_ckpt and latest_ckpt.model_checkpoint_path:
            ck_path = latest_ckpt.model_checkpoint_path
            print("{} found model checkpoint path: {}".format(
                strftime("%H:%M:%S"), ck_path))
            # Extract from checkpoint filename
            bno = int(os.path.basename(ck_path).split('_')[1])
            print('{} resuming from last training, bno = {}'.format(
                strftime("%H:%M:%S"), bno))
            model = keras.models.load_model(ck_path)
            initial_epoch = model.optimizer.iterations.numpy()
            if bno != initial_epoch:
                print(
                    '{} bno({}) from checkpoint file inconsistent with optimizer iterations({}).'
                    .format(strftime("%H:%M:%S"), bno, initial_epoch))
            regressor.setModel(model)
            restored = True
        else:
            print(
                "{} model checkpoint path not found, cleaning training folder".
                format(strftime("%H:%M:%S")))
            tf.io.gfile.rmtree(training_dir)
    else:
        tf.io.gfile.makedirs(training_dir)
        tf.io.gfile.makedirs(best_dir)
        tf.io.gfile.makedirs(log_dir)

    train_ds, test_ds = getInput(bno, args)
    model = regressor.getModel()
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    model.fit(
        x=train_ds,
        verbose=2,
        # must we specify the epochs explicitly?
        epochs=sys.maxsize,
        # Epoch at which to start training (for resuming a previous training run)
        initial_epoch=bno,
        # steps_per_epoch=EVALUATION_INTERVAL, # If None, the epoch will run until the input dataset is exhausted.
        validation_data=test_ds,
        #validation_steps=50,
        validation_freq=VAL_SAVE_FREQ,
        callbacks=callbacks)

    # Export the finalized graph and the variables to the platform-agnostic SavedModel format.
    model.save(filepath=os.path.join(training_dir, 'final.tf'),
               save_format='tf')

    # training finished, move to 'trained' folder
    trained = os.path.join(base_dir, 'trained')
    tf.io.gfile.makedirs(trained)
    tmp_dir = os.path.join(base_dir, strftime("%Y%m%d_%H%M%S"))
    os.rename(training_dir, tmp_dir)
    shutil.move(tmp_dir, trained)
    print('{} model has been saved to {}'.format(strftime("%H:%M:%S"),
                                                 trained))


if __name__ == '__main__':
    args = parseArgs()
    run(args)
