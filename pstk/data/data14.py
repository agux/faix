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
OHLCV-LR + ma-LR(40) + Indicator(KDJ(3),MACD(3),RSI(3),BIAS(3),BOLL(12))
+ SH & SZ indices-LR(69*2) (69*3=207)

Read from backward reinstated klines.

In-place feature-wise standardization is adopted.
'''


k_cols = [
    "lr", "lr_h", "lr_o", "lr_l", "lr_vol",
    "lr_ma5", "lr_ma5_o", "lr_ma5_h", "lr_ma5_l",
    "lr_ma10", "lr_ma10_o", "lr_ma10_h", "lr_ma10_l",
    "lr_ma20", "lr_ma20_o", "lr_ma20_h", "lr_ma20_l",
    "lr_ma30", "lr_ma30_o", "lr_ma30_h", "lr_ma30_l",
    "lr_ma60", "lr_ma60_o", "lr_ma60_h", "lr_ma60_l",
    "lr_ma120", "lr_ma120_o", "lr_ma120_h", "lr_ma120_l",
    "lr_ma200", "lr_ma200_o", "lr_ma200_h", "lr_ma200_l",
    "lr_ma250", "lr_ma250_o", "lr_ma250_h", "lr_ma250_l",
    "lr_vol5", "lr_vol10", "lr_vol20", "lr_vol30",
    "lr_vol60", "lr_vol120", "lr_vol200", "lr_vol250"
]

i_cols = [
    "KDJ_K", "KDJ_D", "KDJ_J",
    "MACD", "MACD_diff", "MACD_dea",
    "RSI1", "RSI2", "RSI3",
    "BIAS1", "BIAS2", "BIAS3",
    "BOLL_lower_o", "BOLL_lower_h", "BOLL_lower_l", "BOLL_lower_c",
    "BOLL_mid_o", "BOLL_mid_h", "BOLL_mid_l", "BOLL_mid_c",
    "BOLL_upper_o", "BOLL_upper_h", "BOLL_upper_l", "BOLL_upper_c"
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
    "    {1} "  # (idc.COLS - mean) / std
    "    {2} "  # COALESCE((sh.COLS - mean) / std, 0) sh_COLS
    "    {3} "  # COALESCE((sz.COLS - mean) / std, 0) sz_COLS
    "FROM "
    "    kline_d_b d "
    "        INNER JOIN "
    "    indicator_d idc USING (code , date , klid) "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        kline_d.date, "
    "        {4} "  # k_cols + i_cols
    "    FROM "
    "        kline_d "
    "    INNER JOIN indicator_d USING (code , date , klid) "
    "    WHERE "
    "        code = 'sh000001') sh USING (date) "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        kline_d.date, "
    "        {4} "  # k_cols + i_cols
    "    FROM "
    "        kline_d "
    "    INNER JOIN indicator_d USING (code , date , klid) "
    "    WHERE "
    "        code = 'sz399001') sz USING (date) "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        %s code, "
    "        t.method, "
    "        {5} "  # mean & std fields
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
    p_indicator = sep.join(["(idc.{0}-s.{0}_mean)/s.{0}_std {0},".format(c)
                            for c in i_cols])
    sh_cols = sep.join(["COALESCE((sh.{0}-s.{0}_mean)/s.{0}_std, 0) sh_{0},".format(c)
                        for c in k_cols+i_cols])
    sz_cols = sep.join(["COALESCE((sz.{0}-s.{0}_mean)/s.{0}_std, 0) sz_{0},".format(c)
                        for c in k_cols+i_cols])
    sz_cols = sz_cols[:-1]  # strip last comma
    all_cols = sep.join(["{},".format(c) for c in k_cols+i_cols])
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
    stats = sep.join([stats_tpl.format(c) for c in k_cols+i_cols])
    stats = stats[:-1]  # strip last comma

    ftQuery = ftQueryTpl.format(
        p_kline, p_indicator, sh_cols, sz_cols, all_cols, stats)
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
