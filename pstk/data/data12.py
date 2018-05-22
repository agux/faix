from __future__ import print_function
import mysql.connector
import sys
import sqlalchemy as sqla
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing
from pprint import pprint
from time import strftime
from joblib import Parallel, delayed
from data import connect

'''
OHLCV-LR + SH & SZ indices-LR(5*2) (5+10=15)
Read from backward reinstated klines.
In-place feature-wise standardization is adopted.
'''

k_cols = [
    "lr", "lr_h", "lr_o", "lr_l", "lr_vol",
]

nclsQry = (
    "SELECT  "
    "    COUNT(*) "
    "FROM "
    "    (SELECT DISTINCT "
    "        score "
    "    FROM "
    "        kpts30) t"
)

ftQueryTpl = (
    "SELECT  "
    "    {0} "  # (d.COLS - mean) / std
    "    {1} "  # COALESCE((sh.COLS - mean) / std, 0) sh_COLS
    "    {2} "  # COALESCE((sz.COLS - mean) / std, 0) sz_COLS
    "FROM "
    "    kline_d_b d "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        kline_d.date, "
    "        {3} "  # k_cols
    "    FROM "
    "        kline_d "
    "    WHERE "
    "        code = 'sh000001') sh USING (date) "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        kline_d.date, "
    "        {3} "  # k_cols
    "    FROM "
    "        kline_d "
    "    WHERE "
    "        code = 'sz399001') sz USING (date) "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        %s code, "
    "        t.method, "
    "        {4} "  # mean & std fields
    "    FROM "
    "        fs_stats t "
    "    WHERE "
    "        t.method = 'standardization' "
    "    GROUP BY code , t.method) s USING (code) "
    "WHERE "
    "    d.code = %s "
    "        AND d.klid BETWEEN %s AND %s "
    "ORDER BY klid "
    "LIMIT %s "
)

num_cores = multiprocessing.cpu_count()

def getFtQuery():
    sep = " "
    p_kline = sep.join(
        ["(d.{0}-s.{0}_mean)/s.{0}_std {0},".format(c) for c in k_cols])
    sh_cols = sep.join(["COALESCE((sh.{0}-s.{0}_mean)/s.{0}_std, 0) sh_{0},".format(c)
                        for c in k_cols])
    sz_cols = sep.join(["COALESCE((sz.{0}-s.{0}_mean)/s.{0}_std, 0) sz_{0},".format(c)
                        for c in k_cols])
    sz_cols = sz_cols[:-1]  # strip last comma
    all_cols = sep.join(["{},".format(c) for c in k_cols])
    all_cols = all_cols[:-1]  # strip last comma
    stats_tpl = (
        " MAX(CASE "
        "     WHEN t.fields = '{0}' THEN t.mean "
        "     ELSE NULL "
        " END) AS {0}_mean, "
        " MAX(CASE "
        "     WHEN t.fields = '{0}' THEN t.std "
        "     ELSE NULL "
        " END) AS {0}_std,"
    )
    stats = sep.join([stats_tpl.format(c) for c in k_cols])
    stats = stats[:-1]  # strip last comma

    ftQuery = ftQueryTpl.format(
        p_kline, sh_cols, sz_cols, all_cols, stats)
    return ftQuery


ftQuery = getFtQuery()


def getBatch(code, s, e, max_step, time_shift):
    '''
    [max_step, feature*time_shift], length
    '''
    cnx = connect()
    fcursor = cnx.cursor(buffered=True)
    global ftQuery
    try:
        fcursor.execute(ftQuery, (code, code, s, e, max_step+time_shift))
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
        cnx.close()


def getSeries(uuid, code, klid, score, nclass, max_step, time_shift):
    label = np.zeros(nclass, dtype=np.int8)
    label[int(score)+nclass//2] = 1
    s = max(0, klid-max_step+1-time_shift)
    batch, total = getBatch(code, s, klid, max_step, time_shift)
    return uuid, batch, label, total

class DataLoader:
    def __init__(self, time_shift):
        self.time_shift = time_shift

    def loadTestSet(self, max_step):
        global num_cores
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
            r = Parallel(n_jobs=num_cores)(delayed(getSeries)(
                uuid, code, klid, score, nclass, max_step, self.time_shift) for uuid, code, klid, score in kpts)
            uuids, data, labels, seqlen = zip(*r)
            # data = [batch, max_step, feature*time_shift]
            # labels = [batch, label]  one-hot labels
            # seqlen = [batch]
            return np.array(uuids), np.array(data), np.array(labels), np.array(seqlen)
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
            r = Parallel(n_jobs=num_cores)(delayed(getSeries)(
                uuid, code, klid, score, nclass, max_step, self.time_shift) for uuid, code, klid, score in kpts)
            uuids, data, labels, seqlen = zip(*r)
            # data = [batch, max_step, feature*time_shift]
            # labels = [batch, label]  one-hot labels
            # seqlen = [batch]
            return np.array(uuids), np.array(data), np.array(labels), np.array(seqlen)
        except:
            print(sys.exc_info()[0])
            raise
        finally:
            cnx.close()
