# from __future__ import absolute_import, division, print_function, unicode_literals

from corl.wc_test.test_runner import run
from corl.model.tf2 import lstm
from time import strftime
# Path hack.
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# pylint: disable-msg=E0401

LAYER_WIDTH = 256
NUM_LSTM_LAYER = 5
NUM_FCN_LAYER = 5

MAX_STEP = 35
TIME_SHIFT = 4
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-5
LR_DECAY_STEPS = 1000
DECAYED_LR_START = 20000
DROPOUT_DECAY_STEPS = 1000
DECAYED_DROPOUT_START = 20000
SEED = 285139

VAL_SAVE_FREQ = 500
STEPS_PER_EPOCH = 500

# feat_cols = ["close", "volume", "amount"]
FEAT_COLS = ["close"]

# pylint: disable-msg=E0601,E1101

if __name__ == '__main__':

    np.random.seed(SEED)

    regressor = lstm.LSTMRegressorV2(
        layer_width=LAYER_WIDTH,
        num_lstm_layer=NUM_LSTM_LAYER,
        num_fcn_layer=NUM_FCN_LAYER,
        time_step=MAX_STEP,
        feat_size=len(FEAT_COLS) * 2 * (TIME_SHIFT + 1),
        dropout_rate=DROPOUT_RATE,
        decayed_dropout_start=DECAYED_DROPOUT_START,
        dropout_decay_steps=DROPOUT_DECAY_STEPS,
        learning_rate=LEARNING_RATE,
        decayed_lr_start=DECAYED_LR_START,
        lr_decay_steps=LR_DECAY_STEPS,
        seed=SEED)
    
    run(
        id="test21_lstm",
        regressor=regressor, 
        max_step=MAX_STEP, 
        time_shift=TIME_SHIFT, 
        feat_cols=FEAT_COLS,
        val_save_freq=VAL_SAVE_FREQ,
        steps_per_epoch=STEPS_PER_EPOCH,
    )
