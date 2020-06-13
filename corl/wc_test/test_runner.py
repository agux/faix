# from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
import random
import shutil
import math
import logging
import ray
import psutil
from pathlib import Path
from corl.wc_test.common import parseArgs, cleanup, LOG_DIR, log, setupPath, DebugCallback
from corl.wc_data import input_fn
from time import strftime
from corl.model.tf2 import lstm
import tensorflow as tf
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# pylint: disable-msg=E0401

VSET = 5
MAX_STEP = 35
TIME_SHIFT = 4

# validate and save the model every n epochs
VAL_SAVE_FREQ = 5
STEPS_PER_EPOCH = 10

# feat_cols = ["close", "volume", "amount"]
FEAT_COLS = ["close"]

# pylint: disable-msg=E0601,E1101

def run(id=None, 
    regressor=None, 
    vset=None, 
    max_step=None, 
    time_shift=None, 
    feat_cols=None, 
    val_save_freq=None, 
    steps_per_epoch=None):

    global VSET, MAX_STEP, TIME_SHIFT, FEAT_COLS, VAL_SAVE_FREQ, STEPS_PER_EPOCH
    VSET = vset or VSET
    MAX_STEP = max_step or MAX_STEP
    TIME_SHIFT = time_shift or TIME_SHIFT
    FEAT_COLS = feat_cols or FEAT_COLS
    VAL_SAVE_FREQ = val_save_freq or VAL_SAVE_FREQ
    STEPS_PER_EPOCH = steps_per_epoch or STEPS_PER_EPOCH

    log.setLevel(logging.WARN)
    args = parseArgs()
    setupPath()
    _setupTensorflow(args)
    _main(args, regressor, id)

def _getInput(start_epoch, args):
    ds = args.ds.lower()
    print('{} using data source: {}'.format(strftime("%H:%M:%S"), args.ds))
    start_bno = start_epoch * STEPS_PER_EPOCH + 1
    input_dict = {}
    if ds == 'db':
        input_dict = input_fn.getInputs(start_bno, TIME_SHIFT, FEAT_COLS,
                                        MAX_STEP, args.parallel, args.prefetch,
                                        args.db_pool, args.db_host,
                                        args.db_port, args.db_pwd, args.vset
                                        or VSET, args.check_input)
    input_dict['start_epoch'] = start_epoch
    return input_dict


def _train(args, regressor, input_dict, base_dir, training_dir):
    # tf.compat.v1.disable_eager_execution()
    print("{} TensorFlow version: {}".format(strftime("%H:%M:%S"),
                                             tf.__version__))
    print("{} Eager execution: {}".format(strftime("%H:%M:%S"),
                                          tf.executing_eagerly()))
    print("{} training started, pid:{}".format(strftime("%H:%M:%S"),
                                               os.getpid()))

    # Define folder paths
    # Define the checkpoint directory to store the checkpoints
    log_dir = os.path.join(base_dir, 'tblogs')
    best_dir = os.path.join(base_dir, 'best')
    if not os.path.exists(training_dir):
        tf.io.gfile.makedirs(training_dir)
    if not os.path.exists(best_dir):
        tf.io.gfile.makedirs(best_dir)
    if not os.path.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)

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
        #Profile the batch to sample compute characteristics. 
        # By default, it will profile the second batch. 
        # Set profile_batch=0 to disable profiling. Must run in TensorFlow eager mode.
        profile_batch=args.profile_batch or 0,
        #the log file can become quite large when write_graph is set to True.
        # write_graph=True,
        #whether to write model weights to visualize as image in TensorBoard.
        # write_images=True,
        
        # 'batch' or 'epoch' or integer.
        # When using 'batch', writes the losses and metrics to TensorBoard after each batch.
        # The same applies for 'epoch'. If using an integer, let's say 1000,
        # the callback will write the metrics and losses to TensorBoard every 1000 batches.
        update_freq='epoch')
    callbacks = [
        # decay,
        tensorboard_cbk,
        DebugCallback() if args.check_weights else None,
        keras.callbacks.CSVLogger('train_perf.log'),
        keras.callbacks.TerminateOnNaN() if args.terminate_on_nan else None,
        # tf.keras.callbacks.ProgbarLogger(count_mode='steps',
        #                                  stateful_metrics=None),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(training_dir,
                                  "cp_{epoch}_{val_loss:.3f}.ckpt"),
            # monitor='val_loss',
            save_weights_only=True,
            verbose=1,
            # 'epoch' or integer. When using 'epoch',
            # the callback saves the model after each epoch.
            # When using integer, the callback saves the model at end of a batch
            # at which this many samples have been seen since last saving.
            save_freq='epoch',
            # period=VAL_SAVE_FREQ
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(best_dir, "cp_best.ckpt"),
            # monitor='val_loss',
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
            save_freq='epoch'
            # period='epoch'
        )
    ]
    callbacks = [c for c in callbacks if c is not None]

    epochs = math.ceil(input_dict['train_batches'] / STEPS_PER_EPOCH)
    # val_freq = list(
    #     chain(range(1, VAL_SAVE_FREQ + 1),
    #           range(VAL_SAVE_FREQ, epochs, VAL_SAVE_FREQ)))
    model = regressor.getModel()
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    model.fit(
        x=input_dict['train'],
        verbose=2,
        # must we specify the epochs explicitly?
        epochs=epochs,
        # Avoids WARNING: Expected a shuffled dataset but input dataset `x` is not shuffled.
        # Please invoke `shuffle()` on input dataset.
        shuffle=False,
        # Epoch at which to start training (for resuming a previous training run)
        initial_epoch=input_dict['start_epoch'],
        # If None, the epoch will run until the input dataset is exhausted.
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=input_dict['test'],
        validation_steps=input_dict['test_batches'],
        # If an integer, specifies how many training epochs to run before a new validation run is performed
        # If a Container, specifies the epochs on which to run validation
        validation_freq=1,
        callbacks=callbacks)

    iterations = model.optimizer.iterations.numpy()
    print('{} Training ended. Finished iterations: {}/{}'.format(
        strftime("%H:%M:%S"), iterations, input_dict['train_batches']))
    if iterations < input_dict['train_batches']:
        return

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


def _load_model(regressor, training_dir):
    # Check if previous training progress exists
    start_epoch = 0
    restored = False
    if os.path.exists(training_dir):
        ckpts = sorted(Path(training_dir).iterdir(), key=os.path.getmtime)
        if len(ckpts) > 0:
            print("{} training folder exists. #checkpoints: {}".format(
                strftime("%H:%M:%S"), len(ckpts)))
            # ck_path = ckpts[-1].resolve()
            ck_path = tf.train.latest_checkpoint(training_dir)
            print("{} latest model checkpoint path: {}".format(
                strftime("%H:%M:%S"), ck_path))
            # Extract from checkpoint filename
            start_epoch = int(os.path.basename(ck_path).split('_')[1])
            print('{} resuming from last training, bno = {}'.format(
                strftime("%H:%M:%S"), start_epoch))
            model = regressor.getModel()
            model.load_weights(str(ck_path))
            regressor.compile()
            restored = True
            iterations = model.optimizer.iterations.numpy()
            initial_epoch = iterations // STEPS_PER_EPOCH
            if start_epoch != initial_epoch:
                print(
                    '{} bno({}) from checkpoint file name inconsistent with saved model({}). optimizer iterations({}).'
                    .format(strftime("%H:%M:%S"), start_epoch, initial_epoch,
                            iterations))
            # else:
            #     print(
            #         "{} model checkpoint path not found, cleaning training folder".
            #         format(strftime("%H:%M:%S")))
            #     tf.io.gfile.rmtree(training_dir)
    if not restored:
        model = regressor.getModel()
        regressor.compile()
    return start_epoch


def _main(args, regressor, id=None):
    model_name = regressor.getName()
    print('{} using model: {}'.format(strftime("%H:%M:%S"), model_name))

    # Define folder paths
    f = __file__
    testn = id or (f[f.rfind('/') + 1:f.rindex('.py')])
    base_name = "{}_{}".format(testn, model_name)
    base_dir = os.path.join(args.logdir or LOG_DIR, base_name)
    training_dir = os.path.join(base_dir, 'training')

    start_epoch = _load_model(regressor, training_dir)

    print('{} querying datasource...'.format(strftime("%H:%M:%S")))
    input_dict = _getInput(start_epoch, args)
    _train(args, regressor, input_dict, base_dir, training_dir)

def _setupTensorflow(args):
    # if args.check_weights:
    #     tf.debugging.enable_check_numerics()
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            print('{} enabling memory growth for {}'.format(strftime("%H:%M:%S"), physical_devices[0]))
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            print(
                'Invalid device or cannot modify virtual devices once initialized.\n'
                + sys.exc_info()[0])
            pass
    
    # enalbe XLA
    tf.config.optimizer.set_jit(True)
    # use mixed precision
    policy  = tf.keras.mixed_precision.experimental.Policy('mixed_float16')   # 'mixed_float16' or 'float32'
    tf.keras.mixed_precision.experimental.set_policy(policy)

