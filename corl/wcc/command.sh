#!/bin/sh

python infer_tf2.py -m [PATH TO THE SAVED MODEL] --gpu_grow_mem --prefetch=2 --parallel=2 --db_host=[DB_HOST] --db_port=[DB_PORT] --db_pwd=[DB_PWD]
if [ $? -eq 0 ]; then
    sleep 15
    echo "job complete, shutting down vm." | mail -s "Job Complete" 3110798@qq.com
else
    echo FAIL
fi

echo 'syncing file system...'
sync
tar -zcvf nohup.tar.gz nohup.log
gsutil cp "nohup.tar.gz" gs://carusytes_bucket/
echo 'shutting down vm...'
sudo shutdown -h now