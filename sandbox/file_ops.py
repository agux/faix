import os
import shutil
import asyncio
from pathlib import Path

#Inputs:
dirpath = '/Users/jx/ProgramData/python/faix/corl/wc_test/logdir/test16_LSTMRegressorV1/training'
keep = 5


async def log_watchdog(dirpath, keep=5, interval=30):
    while True:
        paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)
        # print(paths)
        dir_rm = len(paths) - keep
        if dir_rm == 0:
            print('no folder to delete')
        for i in range(dir_rm):
            print('removing {}'.format(paths[i]))
            shutil.rmtree(paths[i], ignore_errors=True)

        await asyncio.sleep(interval)


async def sleep():
    await asyncio.sleep(10)


async def main():
    t1 = asyncio.create_task(sleep())
    t2 = asyncio.create_task(log_watchdog(dirpath, keep, 2))

    await t1


asyncio.run(main())
