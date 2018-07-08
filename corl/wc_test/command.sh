#!/bin/sh

python test11.py --prefetch=2 --ds=file --dir=gs://carusytes_bucket/wc_train
if [ $? -eq 0 ]; then
    sleep 15
    echo 'syncing file system...'
    sync
    echo 'shutting down vm...'
    sudo shutdown -h now
else
    echo FAIL
fi