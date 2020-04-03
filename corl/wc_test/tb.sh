#!/bin/sh

nohup `tensorboard --bind_all --logdir='$1'`  >tensorboard.log 2>&1` &