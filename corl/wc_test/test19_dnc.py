# from __future__ import absolute_import, division, print_function, unicode_literals

from corl.wc_test.test_runner import run
from corl.model.tf2 import dnc_regressor
from time import strftime
# Path hack.
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# pylint: disable-msg=E0401

CONTROLLER_UNITS = 256
WORD_SIZE = 64
MEMORY_SIZE = 64
NUM_READ_HEADS = 16

MAX_STEP = 35
TIME_SHIFT = 4
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-3
LR_DECAY_STEPS = 1000
CLIP_VALUE = 50
DECAYED_LR_START = 40000
DROPOUT_DECAY_STEPS = 1000
DECAYED_DROPOUT_START = 40000
SEED = 285139

VAL_SAVE_FREQ = 100
STEPS_PER_EPOCH = 100

# feat_cols = ["close", "volume", "amount"]
FEAT_COLS = ["close"]

# pylint: disable-msg=E0601,E1101

if __name__ == '__main__':

    np.random.seed(SEED)

    regressor = dnc_regressor.DNC_Model(
        controller_units=CONTROLLER_UNITS, memory_size=MEMORY_SIZE,
        word_size=WORD_SIZE, num_read_heads=NUM_READ_HEADS,
        time_step=MAX_STEP,
        feat_size=len(FEAT_COLS) * 2 * (TIME_SHIFT + 1),
        dropout_rate=DROPOUT_RATE,
        decayed_dropout_start=DECAYED_DROPOUT_START,
        dropout_decay_steps=DROPOUT_DECAY_STEPS,
        learning_rate=LEARNING_RATE,
        decayed_lr_start=DECAYED_LR_START,
        lr_decay_steps=LR_DECAY_STEPS,
        clipvalue=CLIP_VALUE
    )
    
    run(
        id="test19",
        regressor=regressor, 
        max_step=MAX_STEP, 
        time_shift=TIME_SHIFT, 
        feat_cols=FEAT_COLS,
        val_save_freq=VAL_SAVE_FREQ,
        steps_per_epoch=STEPS_PER_EPOCH,
    )
