from __future__ import print_function
import mysql.connector
import sys
import sqlalchemy as sqla
import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint
from time import strftime

from data import connect

'''
OHLCV-LR + SH & SZ indices-LR(5*2) (5+10=15)
Read from backward reinstated klines.
Standardization is adopted.
'''

TIME_SHIFT = 9

nclsQry = (
    "SELECT  "
    "    COUNT(*) "
    "FROM "
    "    (SELECT DISTINCT "
    "        score "
    "    FROM "
    "        kpts30) t"
)

ftQuery = (
    "SELECT  "
    "    d.lr, "
    "    d.lr_h, "
    "    d.lr_o, "
    "    d.lr_l, "
    "    d.lr_vol, "
    "    COALESCE(sh.lr,0) sh_lr, "
    "    COALESCE(sh.lr_h,0) sh_lr_h, "
    "    COALESCE(sh.lr_o,0) sh_lr_o, "
    "    COALESCE(sh.lr_l,0) sh_lr_l, "
    "    COALESCE(sh.lr_vol,0) sh_lr_vol, "
    "    COALESCE(sz.lr,0) sz_lr, "
    "    COALESCE(sz.lr_h,0) sz_h, "
    "    COALESCE(sz.lr_o,0) sz_o, "
    "    COALESCE(sz.lr_l,0) sz_l, "
    "    COALESCE(sz.lr_vol,0) sz_vol "
    "FROM "
    "    kline_d_b d "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        lr, lr_h, lr_o, lr_l, lr_vol, date "
    "    FROM "
    "        kline_d "
    "    WHERE "
    "        code = 'sh000001') sh USING (date) "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        lr, lr_h, lr_o, lr_l, lr_vol, date "
    "    FROM "
    "        kline_d "
    "    WHERE "
    "        code = 'sz399001') sz USING (date) "
    "WHERE "
    "    d.code = %s "
    "        AND d.klid BETWEEN %s AND %s "
    "ORDER BY klid "
    "LIMIT %s "
)

mean = None
std = None


def loadTestSet(max_step):
    cnx = connect()
    try:
        nc_cursor = cnx.cursor(buffered=True)
        nc_cursor.execute(nclsQry)
        row = nc_cursor.fetchone()
        nclass = int(row[0])
        print('{} num class: {}'.format(strftime("%H:%M:%S"), nclass))
        nc_cursor.close()
        cursor = cnx.cursor(buffered=True)
        pick = (
            "SELECT  "
            "    distinct flag "
            "FROM "
            "    kpts30 "
            "WHERE "
            "    flag LIKE 'TEST\\_%' "
            "ORDER BY RAND() "
            "LIMIT 1"
        )
        cursor.execute(pick)
        row = cursor.fetchone()
        print('{} selected test set: {}'.format(strftime("%H:%M:%S"), row[0]))
        query = (
            "SELECT "
            "   uuid, code, klid, score "
            "FROM "
            "   kpts30 "
            "WHERE "
            "   flag = '{}' "
        )
        cursor.execute(query.format(row[0]))
        kpts = cursor.fetchall()
        cursor.close()
        data = []   # [batch, max_step, feature*time_shift]
        labels = []  # [batch, label]  one-hot labels
        seqlen = []  # [batch]
        uuids = []
        for (uuid, code, klid, score) in kpts:
            uuids.append(uuid)
            label = np.zeros(nclass, dtype=np.int8)
            label[int(score)+nclass//2] = 1
            labels.append(label)
            s = max(0, klid-max_step+1-TIME_SHIFT)
            batch, total = getBatch(cnx, code, s, klid, max_step)
            data.append(batch)
            seqlen.append(total)
        return uuids, np.array(data), np.array(labels), np.array(seqlen)
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


def getStats(cnx):
    global mean, std
    if mean is not None and std is not None:
        return mean, std
    c = cnx.cursor(buffered=True)
    c.execute("select mean, `std` from fs_stats where method = 'standardization'")
    row = c.fetchone()
    mean, std = row[0], row[1]
    print("mean: {}, std: {}".format(mean, std))
    return mean, std


def getBatch(cnx, code, s, e, max_step):
    '''
    [max_step, feature*time_shift], length
    '''
    fcursor = cnx.cursor(buffered=True)
    try:
        fcursor.execute(ftQuery, (code, s, e, max_step+TIME_SHIFT))
        featSize = len(fcursor.column_names)
        total = fcursor.rowcount
        rows = fcursor.fetchall()
        batch = []
        # mean, std = getStats(cnx)
        for t in range(TIME_SHIFT+1):
            steps = np.zeros((max_step, featSize), dtype='f')
            offset = max_step + TIME_SHIFT - total
            s = max(0, t - offset)
            e = total - TIME_SHIFT + t
            for i, row in enumerate(rows[s:e]):
                steps[i+offset] = [col for col in row]
                # steps[i+offset] = [(col-mean)/std for col in row]
            batch.append(steps)
        return np.concatenate(batch, 1), total - TIME_SHIFT
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        fcursor.close()


def loadTrainingData(batch_no, max_step):
    cnx = connect()
    try:
        nc_cursor = cnx.cursor(buffered=True)
        nc_cursor.execute(nclsQry)
        row = nc_cursor.fetchone()
        nclass = int(row[0])
        nc_cursor.close()
        cursor = cnx.cursor(buffered=True)
        query = (
            'SELECT '
            '   uuid, code, klid, score '
            'FROM'
            '   kpts30 '
            'WHERE '
            "   flag = 'TRN_{}'"
        )
        # print(query)
        cursor.execute(query.format(batch_no))
        kpts = cursor.fetchall()
        cursor.close()
        data = []   # [batch, max_step, feature*time_shift]
        labels = []  # [batch, label]  one-hot labels
        seqlen = []  # [batch]
        uuids = []
        for (uuid, code, klid, score) in kpts:
            uuids.append(uuid)
            label = np.zeros((nclass), dtype=np.int8)
            label[int(score)+nclass//2] = 1
            labels.append(label)
            s = max(0, klid-max_step+1-TIME_SHIFT)
            batch, total = getBatch(cnx, code, s, klid, max_step)
            data.append(batch)
            seqlen.append(total)
        # pprint(data)
        # print("\n")
        # pprint(len(labels))
        return uuids, np.array(data), np.array(labels), np.array(seqlen)
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()
