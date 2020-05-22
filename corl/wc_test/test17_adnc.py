# from __future__ import absolute_import, division, print_function, unicode_literals

from corl.wc_test.test_runner import run
from corl.model.tf2 import dnc
from time import strftime
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# pylint: disable-msg=E0401

VSET = 5
LAYER_WIDTH = 256
MAX_STEP = 35
TIME_SHIFT = 19
DROPOUT_RATE = 0.5
LEARNING_RATE = 5e-5
LR_DECAY_STEPS = 1000
DECAYED_LR_START = 40000
DROPOUT_DECAY_STEPS = 1000
DECAYED_DROPOUT_START = 40000
SEED = 285139

# validate and save the model every n epochs
VAL_SAVE_FREQ = 5
STEPS_PER_EPOCH = 10

# feat_cols = ["close", "volume", "amount"]
feat_cols = ["close"]

# pylint: disable-msg=E0601,E1101

if __name__ == '__main__':

    controller_config = {
        "num_units":[128], 
        "layer_norm":True, 
        "activation":'tanh', 
        'cell_type':'clstm', 
        'connect':'sparse',
    }

    memory_unit_config = {
        "cell_type":'cbmu', 
        "memory_length":64, 
        "memory_width":32, 
        "read_heads":4, 
        "write_heads": 2, 
        "dnc_norm":True, 
        "bypass_dropout":False, 
        "wgate1":False,
    }

    regressor = dnc.MANN_Model(
        controller_config,
        memory_unit_config,
        time_step=MAX_STEP,
        feat_size=len(feat_cols) * 2 * (TIME_SHIFT + 1),
        dropout_rate=DROPOUT_RATE,
        decayed_dropout_start=DECAYED_DROPOUT_START,
        dropout_decay_steps=DROPOUT_DECAY_STEPS,
        learning_rate=LEARNING_RATE,
        decayed_lr_start=DECAYED_LR_START,
        lr_decay_steps=LR_DECAY_STEPS,
        seed=SEED)
    
    run(regressor)
