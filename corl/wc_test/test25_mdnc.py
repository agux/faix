# from __future__ import absolute_import, division, print_function, unicode_literals

from corl.wc_test.test_runner import run
from corl.model.tf2 import dnc_regressor
from corl.wc_test.common import next_power_of_2
from time import strftime
# Path hack.
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# pylint: disable-msg=E0401

MAX_STEP = 35
TIME_SHIFT = 4
DROPOUT_RATE = 0
LEARNING_RATE = 1e-4
LR_DECAY_STEPS = 1000
CLIP_VALUE = 50
DECAYED_LR_START = 20000
DROPOUT_DECAY_STEPS = 1000
DECAYED_DROPOUT_START = 20000
SEED = 285139

FEAT_COLS = ["close"]
FEAT_SIZE = len(FEAT_COLS) * 2 * (TIME_SHIFT + 1)

NUM_CNN_LAYERS = 3
NUM_DNC_LAYERS = 3
NUM_FCN_LAYERS = 0
CNN_FILTERS = next_power_of_2(FEAT_SIZE * 2)
CNN_KERNEL_SIZE = TIME_SHIFT
LAYER_NORM_LSTM = True
CONTROLLER_UNITS = CNN_FILTERS // 2
DNC_OUTPUT_SIZE = CNN_FILTERS
WORD_SIZE = 32
MEMORY_SIZE = 32
NUM_READ_HEADS = 8

VAL_SAVE_FREQ = 500
STEPS_PER_EPOCH = 500

INCLUDE_SEQLENS = False
# feat_cols = ["close", "volume", "amount"]

# pylint: disable-msg=E0601,E1101

if __name__ == '__main__':

    np.random.seed(SEED)

    regressor = dnc_regressor.DNC_Model_V6(
        num_cnn_layers=NUM_CNN_LAYERS, 
        num_dnc_layers = NUM_DNC_LAYERS,
        num_fcn_layers = NUM_FCN_LAYERS,
        cnn_filters=CNN_FILTERS,
        cnn_kernel_size=CNN_KERNEL_SIZE, #can be a list
        layer_norm_lstm = LAYER_NORM_LSTM,
        output_size=DNC_OUTPUT_SIZE,
        controller_units=CONTROLLER_UNITS, 
        memory_size=MEMORY_SIZE,
        word_size=WORD_SIZE, 
        num_read_heads=NUM_READ_HEADS,
        time_step=MAX_STEP,
        feat_size=FEAT_SIZE,
        dropout_rate=DROPOUT_RATE,
        decayed_dropout_start=DECAYED_DROPOUT_START,
        dropout_decay_steps=DROPOUT_DECAY_STEPS,
        learning_rate=LEARNING_RATE,
        decayed_lr_start=DECAYED_LR_START,
        lr_decay_steps=LR_DECAY_STEPS,
        clipvalue=CLIP_VALUE
    )
    
    run(
        id="test25_mdnc",
        regressor=regressor, 
        max_step=MAX_STEP, 
        time_shift=TIME_SHIFT, 
        feat_cols=FEAT_COLS,
        val_save_freq=VAL_SAVE_FREQ,
        steps_per_epoch=STEPS_PER_EPOCH,
        include_seqlens=INCLUDE_SEQLENS,
    )
