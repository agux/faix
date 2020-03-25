#!/bin/sh

python3 test16.py --prefetch=2 --ds=db --db_pool=4 --db_host=['replace'] --db_port=['replace'] --db_pwd=['replace']
if [ $? -eq 0 ]; then
    sleep 15
    #echo "Training complete, shutting down vm." | mail -s "Training Complete" 3110798@qq.com
    #echo 'syncing file system...'
    sync
    #echo 'shutting down vm...'
    #sudo shutdown -h now
else
    echo FAIL
fi