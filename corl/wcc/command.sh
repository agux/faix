#!/bin/sh

python wcc_infer.py --prefetch=5 -m [model destination] -r gs://carusytes_bucket/wcc_infer -p gs://carusytes_bucket/wcc_infer_results
if [ $? -eq 0 ]; then
    sleep 15
    echo "job complete, shutting down vm." | mail -s "Job Complete" 3110798@qq.com
    echo 'syncing file system...'
    sync
    echo 'shutting down vm...'
    sudo shutdown -h now
else
    echo FAIL
fi