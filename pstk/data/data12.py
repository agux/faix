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
'''


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
    "    COALESCE(sz.lr_h,0) sz_lr_h, "
    "    COALESCE(sz.lr_o,0) sz_lr_o, "
    "    COALESCE(sz.lr_l,0) sz_lr_l, "
    "    COALESCE(sz.lr_vol,0) sz_lr_vol "
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


def getBatch(cnx, code, s, e, max_step, time_shift):
    '''
    [max_step, feature*time_shift], length
    '''
    fcursor = cnx.cursor(buffered=True)
    try:
        fcursor.execute(ftQuery, (code, s, e, max_step+time_shift))
        col_names = fcursor.column_names
        featSize = len(col_names)
        total = fcursor.rowcount
        rows = fcursor.fetchall()
        batch = []
        for t in range(time_shift+1):
            steps = np.zeros((max_step, featSize), dtype='f')
            offset = max_step + time_shift - total
            s = max(0, t - offset)
            e = total - time_shift + t
            for i, row in enumerate(rows[s:e]):
                steps[i+offset] = [col for col in row]
            batch.append(steps)
        return np.concatenate(batch, 1), total - time_shift
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        fcursor.close()


class DataLoader:
    def __init__(self, time_shift):
        self.time_shift = time_shift

    def loadTestSet(self, max_step):
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
            print('{} selected test set: {}'.format(
                strftime("%H:%M:%S"), row[0]))
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
                s = max(0, klid-max_step+1-self.time_shift)
                batch, total = getBatch(
                    cnx, code, s, klid, max_step, self.time_shift)
                data.append(batch)
                seqlen.append(total)
            return uuids, np.array(data), np.array(labels), np.array(seqlen)
        except:
            print(sys.exc_info()[0])
            raise
        finally:
            cnx.close()

    def loadTrainingData(self, batch_no, max_step):
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
                s = max(0, klid-max_step+1-self.time_shift)
                batch, total = getBatch(
                    cnx, code, s, klid, max_step, self.time_shift)
                data.append(batch)
                seqlen.append(total)
            return uuids, np.array(data), np.array(labels), np.array(seqlen)
        except:
            print(sys.exc_info()[0])
            raise
        finally:
            cnx.close()
