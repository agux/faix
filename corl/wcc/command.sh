#!/bin/sh

python wcc_infer.py --prefetch=5 -m [model destination] -r gs://carusytes_bucket/wcc_infer -p gs://carusytes_bucket/wcc_infer_results
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