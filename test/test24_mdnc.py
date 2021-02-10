
from common.train_runner import run
from model import dnc_regressor
from common.common import next_power_of_2
# Path hack.
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# pylint: disable-msg=E0401

MAX_STEP = 35
TIME_SHIFT = 4
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-3
CLIP_VALUE = 50
LR_DECAY_STEPS = 2500
DECAYED_LR_START = 40000
DROPOUT_DECAY_STEPS = 2500
DECAYED_DROPOUT_START = 40000
SEED = 285139

FEAT_COLS = ["close"]
FEAT_SIZE = len(FEAT_COLS) * 2 * (TIME_SHIFT + 1)

NUM_CNN_LAYERS = 3
NUM_DNC_LAYERS = 2
NUM_FCN_LAYERS = 3
CNN_FILTERS = next_power_of_2(FEAT_SIZE)
CNN_KERNEL_SIZE = TIME_SHIFT if TIME_SHIFT > 0 else MAX_STEP // 10
CNN_OUTPUT_SIZE = 512
LAYER_NORM_LSTM = True
CONTROLLER_UNITS = 256
DNC_OUTPUT_SIZE = [256, 256]
WORD_SIZE = 32
MEMORY_SIZE = 32
NUM_READ_HEADS = 8

NUM_CLASSES = 5

VAL_SAVE_FREQ = 500
STEPS_PER_EPOCH = 500

INCLUDE_SEQLENS = False
# feat_cols = ["close", "volume", "amount"]

# pylint: disable-msg=E0601,E1101


def create_regressor():
    regressor = dnc_regressor.BaseModel(
        num_cnn_layers=NUM_CNN_LAYERS,
        num_dnc_layers=NUM_DNC_LAYERS,
        num_fcn_layers=NUM_FCN_LAYERS,
        cnn_filters=CNN_FILTERS,
        cnn_kernel_size=CNN_KERNEL_SIZE,
        cnn_output_size=CNN_OUTPUT_SIZE,
        layer_norm_lstm=LAYER_NORM_LSTM,
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
        clipvalue=CLIP_VALUE,
        num_classes=NUM_CLASSES,
        seed=SEED,
    )
    return regressor


if __name__ == '__main__':

    np.random.seed(SEED)

    regressor = create_regressor()

    run(id="stock_trend_test24_mdnc",
        regressor=regressor,
        vset=None,
        max_step=MAX_STEP,
        time_shift=TIME_SHIFT,
        feat_cols=FEAT_COLS,
        val_save_freq=VAL_SAVE_FREQ,
        steps_per_epoch=STEPS_PER_EPOCH,
        data_pipeline=None)
