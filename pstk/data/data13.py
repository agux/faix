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
OHLCV-LR + ma-LR(40) + Indicator(KDJ(3),MACD(3),RSI(3),BIAS(3),BOLL(12))
+ SH & SZ indices-LR(69*2) (69*3=207)

Read from backward reinstated klines.

Feature-wise standardization is adopted.
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
    "    d.lr, d.lr_h, d.lr_o, d.lr_l, d.lr_vol, "
    "    d.lr_ma5, d.lr_ma5_o, d.lr_ma5_h, d.lr_ma5_l, "
    "    d.lr_ma10, d.lr_ma10_o, d.lr_ma10_h, d.lr_ma10_l, "
    "    d.lr_ma20, d.lr_ma20_o, d.lr_ma20_h, d.lr_ma20_l, "
    "    d.lr_ma30, d.lr_ma30_o, d.lr_ma30_h, d.lr_ma30_l, "
    "    d.lr_ma60, d.lr_ma60_o, d.lr_ma60_h, d.lr_ma60_l, "
    "    d.lr_ma120, d.lr_ma120_o, d.lr_ma120_h, d.lr_ma120_l, "
    "    d.lr_ma200, d.lr_ma200_o, d.lr_ma200_h, d.lr_ma200_l, "
    "    d.lr_ma250, d.lr_ma250_o, d.lr_ma250_h, d.lr_ma250_l, "
    "    d.lr_vol5, d.lr_vol10, d.lr_vol20, d.lr_vol30, "
    "    d.lr_vol60, d.lr_vol120, d.lr_vol200, d.lr_vol250, "
    "    idc.KDJ_K, idc.KDJ_D, idc.KDJ_J, "
    "    idc.MACD, idc.MACD_diff, idc.MACD_dea, "
    "    idc.RSI1, idc.RSI2, idc.RSI3,  "
    "    idc.BIAS1, idc.BIAS2, idc.BIAS3,  "
    "    idc.BOLL_lower_o, idc.BOLL_lower_h, idc.BOLL_lower_l, idc.BOLL_lower_c,  "
    "    idc.BOLL_mid_o, idc.BOLL_mid_h, idc.BOLL_mid_l, idc.BOLL_mid_c, "
    "    idc.BOLL_upper_o, idc.BOLL_upper_h, idc.BOLL_upper_l, idc.BOLL_upper_c, "
    "    COALESCE(sh.lr, 0) sh_lr, COALESCE(sh.lr_h, 0) sh_lr_h, COALESCE(sh.lr_o, 0) sh_lr_o, COALESCE(sh.lr_l, 0) sh_lr_l, COALESCE(sh.lr_vol, 0) sh_lr_vol, "
    "    COALESCE(sh.lr_ma5, 0) sh_lr_ma5, COALESCE(sh.lr_ma5_o, 0) sh_lr_ma5_o, COALESCE(sh.lr_ma5_h, 0) sh_lr_ma5_h, COALESCE(sh.lr_ma5_l, 0) sh_lr_ma5_l, "
    "    COALESCE(sh.lr_ma10, 0) sh_lr_ma10, COALESCE(sh.lr_ma10_o, 0) sh_lr_ma10_o, COALESCE(sh.lr_ma10_h, 0) sh_lr_ma10_h, COALESCE(sh.lr_ma10_l, 0) sh_lr_ma10_l, "
    "    COALESCE(sh.lr_ma20, 0) sh_lr_ma20, COALESCE(sh.lr_ma20_o, 0) sh_lr_ma20_o, COALESCE(sh.lr_ma20_h, 0) sh_lr_ma20_h, COALESCE(sh.lr_ma20_l, 0) sh_lr_ma20_l, "
    "    COALESCE(sh.lr_ma30, 0) sh_lr_ma30, COALESCE(sh.lr_ma30_o, 0) sh_lr_ma30_o, COALESCE(sh.lr_ma30_h, 0) sh_lr_ma30_h, COALESCE(sh.lr_ma30_l, 0) sh_lr_ma30_l, "
    "    COALESCE(sh.lr_ma60, 0) sh_lr_ma60, COALESCE(sh.lr_ma60_o, 0) sh_lr_ma60_o, COALESCE(sh.lr_ma60_h, 0) sh_lr_ma60_h, COALESCE(sh.lr_ma60_l, 0) sh_lr_ma60_l, "
    "    COALESCE(sh.lr_ma120, 0) sh_lr_ma120, COALESCE(sh.lr_ma120_o, 0) sh_lr_ma120_o, COALESCE(sh.lr_ma120_h, 0) sh_lr_ma120_h, COALESCE(sh.lr_ma120_l, 0) sh_lr_ma120_l, "
    "    COALESCE(sh.lr_ma200, 0) sh_lr_ma200, COALESCE(sh.lr_ma200_o, 0) sh_lr_ma200_o, COALESCE(sh.lr_ma200_h, 0) sh_lr_ma200_h, COALESCE(sh.lr_ma200_l, 0) sh_lr_ma200_l, "
    "    COALESCE(sh.lr_ma250, 0) sh_lr_ma250, COALESCE(sh.lr_ma250_o, 0) sh_lr_ma250_o, COALESCE(sh.lr_ma250_h, 0) sh_lr_ma250_h, COALESCE(sh.lr_ma250_l, 0) sh_lr_ma250_l, "
    "    COALESCE(sh.lr_vol5, 0) sh_lr_vol5, COALESCE(sh.lr_vol10, 0) sh_lr_vol10, COALESCE(sh.lr_vol20, 0) sh_lr_vol20, COALESCE(sh.lr_vol30, 0) sh_lr_vol30, "
    "    COALESCE(sh.lr_vol60, 0) sh_lr_vol60, COALESCE(sh.lr_vol120, 0) sh_lr_vol120, COALESCE(sh.lr_vol200, 0) sh_lr_vol200, COALESCE(sh.lr_vol250, 0) sh_lr_vol250, "
    "    COALESCE(sh.KDJ_K, 0) sh_KDJ_K, COALESCE(sh.KDJ_D, 0) sh_KDJ_D, COALESCE(sh.KDJ_J, 0) sh_KDJ_J,  "
    "    COALESCE(sh.MACD, 0) sh_MACD, COALESCE(sh.MACD_diff, 0) sh_MACD_diff, COALESCE(sh.MACD_dea, 0) sh_MACD_dea, "
    "    COALESCE(sh.RSI1, 0) sh_RSI1, COALESCE(sh.RSI2, 0) sh_RSI2, COALESCE(sh.RSI3, 0) sh_RSI3, "
    "    COALESCE(sh.BIAS1, 0) sh_BIAS1, COALESCE(sh.BIAS2, 0) sh_BIAS2, COALESCE(sh.BIAS3, 0) sh_BIAS3, "
    "    COALESCE(sh.BOLL_lower_o, 0) sh_BOLL_lower_o, COALESCE(sh.BOLL_lower_h, 0) sh_BOLL_lower_h, COALESCE(sh.BOLL_lower_l, 0) sh_BOLL_lower_l, COALESCE(sh.BOLL_lower_c, 0) sh_BOLL_lower_c, "
    "    COALESCE(sh.BOLL_mid_o, 0) sh_BOLL_mid_o, COALESCE(sh.BOLL_mid_h, 0) sh_BOLL_mid_h, COALESCE(sh.BOLL_mid_l, 0) sh_BOLL_mid_l, COALESCE(sh.BOLL_mid_c, 0) sh_BOLL_mid_c, "
    "    COALESCE(sh.BOLL_upper_o, 0) sh_BOLL_upper_o, COALESCE(sh.BOLL_upper_h, 0) sh_BOLL_upper_h, COALESCE(sh.BOLL_upper_l, 0) sh_BOLL_upper_l, COALESCE(sh.BOLL_upper_c, 0) sh_BOLL_upper_c, "
    "    COALESCE(sz.lr, 0) sz_lr, COALESCE(sz.lr_h, 0) sz_lr_h, COALESCE(sz.lr_o, 0) sz_lr_o, COALESCE(sz.lr_l, 0) sz_lr_l, COALESCE(sz.lr_vol, 0) sz_lr_vol, "
    "    COALESCE(sz.lr_ma5, 0) sz_lr_ma5, COALESCE(sz.lr_ma5_o, 0) sz_lr_ma5_o, COALESCE(sz.lr_ma5_h, 0) sz_lr_ma5_h, COALESCE(sz.lr_ma5_l, 0) sz_lr_ma5_l, "
    "    COALESCE(sz.lr_ma10, 0) sz_lr_ma10, COALESCE(sz.lr_ma10_o, 0) sz_lr_ma10_o, COALESCE(sz.lr_ma10_h, 0) sz_lr_ma10_h, COALESCE(sz.lr_ma10_l, 0) sz_lr_ma10_l, "
    "    COALESCE(sz.lr_ma20, 0) sz_lr_ma20, COALESCE(sz.lr_ma20_o, 0) sz_lr_ma20_o, COALESCE(sz.lr_ma20_h, 0) sz_lr_ma20_h, COALESCE(sz.lr_ma20_l, 0) sz_lr_ma20_l, "
    "    COALESCE(sz.lr_ma30, 0) sz_lr_ma30, COALESCE(sz.lr_ma30_o, 0) sz_lr_ma30_o, COALESCE(sz.lr_ma30_h, 0) sz_lr_ma30_h, COALESCE(sz.lr_ma30_l, 0) sz_lr_ma30_l, "
    "    COALESCE(sz.lr_ma60, 0) sz_lr_ma60, COALESCE(sz.lr_ma60_o, 0) sz_lr_ma60_o, COALESCE(sz.lr_ma60_h, 0) sz_lr_ma60_h, COALESCE(sz.lr_ma60_l, 0) sz_lr_ma60_l, "
    "    COALESCE(sz.lr_ma120, 0) sz_lr_ma120, COALESCE(sz.lr_ma120_o, 0) sz_lr_ma120_o, COALESCE(sz.lr_ma120_h, 0) sz_lr_ma120_h, COALESCE(sz.lr_ma120_l, 0) sz_lr_ma120_l, "
    "    COALESCE(sz.lr_ma200, 0) sz_lr_ma200, COALESCE(sz.lr_ma200_o, 0) sz_lr_ma200_o, COALESCE(sz.lr_ma200_h, 0) sz_lr_ma200_h, COALESCE(sz.lr_ma200_l, 0) sz_lr_ma200_l, "
    "    COALESCE(sz.lr_ma250, 0) sz_lr_ma250, COALESCE(sz.lr_ma250_o, 0) sz_lr_ma250_o, COALESCE(sz.lr_ma250_h, 0) sz_lr_ma250_h, COALESCE(sz.lr_ma250_l, 0) sz_lr_ma250_l, "
    "    COALESCE(sz.lr_vol5, 0) sz_lr_vol5, COALESCE(sz.lr_vol10, 0) sz_lr_vol10, COALESCE(sz.lr_vol20, 0) sz_lr_vol20, COALESCE(sz.lr_vol30, 0) sz_lr_vol30, "
    "    COALESCE(sz.lr_vol60, 0) sz_lr_vol60, COALESCE(sz.lr_vol120, 0) sz_lr_vol120, COALESCE(sz.lr_vol200, 0) sz_lr_vol200, COALESCE(sz.lr_vol250, 0) sz_lr_vol250, "
    "    COALESCE(sz.KDJ_K, 0) sz_KDJ_K, COALESCE(sz.KDJ_D, 0) sz_KDJ_D, COALESCE(sz.KDJ_J, 0) sz_KDJ_J, "
    "    COALESCE(sz.MACD, 0) sz_MACD, COALESCE(sz.MACD_diff, 0) sz_MACD_diff, COALESCE(sz.MACD_dea, 0) sz_MACD_dea, "
    "    COALESCE(sz.RSI1, 0) sz_RSI1, COALESCE(sz.RSI2, 0) sz_RSI2, COALESCE(sz.RSI3, 0) sz_RSI3, "
    "    COALESCE(sz.BIAS1, 0) sz_BIAS1, COALESCE(sz.BIAS2, 0) sz_BIAS2, COALESCE(sz.BIAS3, 0) sz_BIAS3, "
    "    COALESCE(sz.BOLL_lower_o, 0) sz_BOLL_lower_o, COALESCE(sz.BOLL_lower_h, 0) sz_BOLL_lower_h, COALESCE(sz.BOLL_lower_l, 0) sz_BOLL_lower_l, COALESCE(sz.BOLL_lower_c, 0) sz_BOLL_lower_c, "
    "    COALESCE(sz.BOLL_mid_o, 0) sz_BOLL_mid_o, COALESCE(sz.BOLL_mid_h, 0) sz_BOLL_mid_h, COALESCE(sz.BOLL_mid_l, 0) sz_BOLL_mid_l, COALESCE(sz.BOLL_mid_c, 0) sz_BOLL_mid_c, "
    "    COALESCE(sz.BOLL_upper_o, 0) sz_BOLL_upper_o, COALESCE(sz.BOLL_upper_h, 0) sz_BOLL_upper_h, COALESCE(sz.BOLL_upper_l, 0) sz_BOLL_upper_l, COALESCE(sz.BOLL_upper_c, 0) sz_BOLL_upper_c "
    "FROM "
    "    kline_d_b d "
    "        INNER JOIN "
    "    indicator_d idc USING (code , date , klid) "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        kline_d.date, "
    "        lr,lr_h,lr_o,lr_l,lr_vol, "
    "        lr_ma5,lr_ma5_o,lr_ma5_h,lr_ma5_l, "
    "        lr_ma10,lr_ma10_o,lr_ma10_h,lr_ma10_l, "
    "        lr_ma20,lr_ma20_o,lr_ma20_h,lr_ma20_l, "
    "        lr_ma30,lr_ma30_o,lr_ma30_h,lr_ma30_l, "
    "        lr_ma60,lr_ma60_o,lr_ma60_h,lr_ma60_l, "
    "        lr_ma120,lr_ma120_o,lr_ma120_h,lr_ma120_l, "
    "        lr_ma200,lr_ma200_o,lr_ma200_h,lr_ma200_l, "
    "        lr_ma250,lr_ma250_o,lr_ma250_h,lr_ma250_l, "
    "        lr_vol5,lr_vol10,lr_vol20,lr_vol30, "
    "        lr_vol60,lr_vol120,lr_vol200,lr_vol250, "
    "        KDJ_K,KDJ_D,KDJ_J, "
    "        MACD,MACD_diff,MACD_dea, "
    "        RSI1,RSI2,RSI3, "
    "        BIAS1,BIAS2,BIAS3, "
    "        BOLL_lower_o,BOLL_lower_h,BOLL_lower_l,BOLL_lower_c, "
    "        BOLL_mid_o,BOLL_mid_h,BOLL_mid_l,BOLL_mid_c, "
    "        BOLL_upper_o,BOLL_upper_h,BOLL_upper_l,BOLL_upper_c "
    "    FROM "
    "        kline_d "
    "    INNER JOIN indicator_d USING (code , date , klid) "
    "    WHERE "
    "        code = 'sh000001') sh USING (date) "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        kline_d.date, "
    "        lr,lr_h,lr_o,lr_l,lr_vol, "
    "        lr_ma5,lr_ma5_o,lr_ma5_h,lr_ma5_l, "
    "        lr_ma10,lr_ma10_o,lr_ma10_h,lr_ma10_l, "
    "        lr_ma20,lr_ma20_o,lr_ma20_h,lr_ma20_l, "
    "        lr_ma30,lr_ma30_o,lr_ma30_h,lr_ma30_l, "
    "        lr_ma60,lr_ma60_o,lr_ma60_h,lr_ma60_l, "
    "        lr_ma120,lr_ma120_o,lr_ma120_h,lr_ma120_l, "
    "        lr_ma200,lr_ma200_o,lr_ma200_h,lr_ma200_l, "
    "        lr_ma250,lr_ma250_o,lr_ma250_h,lr_ma250_l, "
    "        lr_vol5,lr_vol10,lr_vol20,lr_vol30, "
    "        lr_vol60,lr_vol120,lr_vol200,lr_vol250, "
    "        KDJ_K,KDJ_D,KDJ_J, "
    "        MACD,MACD_diff,MACD_dea, "
    "        RSI1,RSI2,RSI3, "
    "        BIAS1,BIAS2,BIAS3, "
    "        BOLL_lower_o,BOLL_lower_h,BOLL_lower_l,BOLL_lower_c, "
    "        BOLL_mid_o,BOLL_mid_h,BOLL_mid_l,BOLL_mid_c, "
    "        BOLL_upper_o,BOLL_upper_h,BOLL_upper_l,BOLL_upper_c "
    "    FROM "
    "        kline_d "
    "    INNER JOIN indicator_d USING (code , date , klid) "
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


def getStats(cnx):
    global mean, std
    if mean is not None and std is not None:
        return mean, std
    mean = {}
    std = {}
    c = cnx.cursor(buffered=True)
    c.execute(
        "select fields, mean, `std` from fs_stats where method = 'standardization'")
    rows = c.fetchall()
    for (field, m, s) in rows:
        mean["sz_"+field] = mean["sh_"+field] = mean[field] = m
        std["sz_"+field] = std["sh_"+field] = std[field] = s
        print("{} mean: {}, std: {}".format(field, m, s))
    return mean, std


def getBatch(cnx, code, s, e, max_step, time_shift):
    '''
    [max_step, feature*time_shift], length
    '''
    fcursor = cnx.cursor(buffered=True, dictionary=True)
    try:
        fcursor.execute(ftQuery, (code, s, e, max_step+time_shift))
        col_names = fcursor.column_names
        featSize = len(col_names)
        total = fcursor.rowcount
        rows = fcursor.fetchall()
        batch = []
        mean, std = getStats(cnx)
        for t in range(time_shift+1):
            steps = np.zeros((max_step, featSize), dtype='f')
            offset = max_step + time_shift - total
            s = max(0, t - offset)
            e = total - time_shift + t
            for i, row in enumerate(rows[s:e]):
                # steps[i+offset] = [col for col in row]
                steps[i+offset] = [(val-mean[col])/std[col]
                                   for col, val in row.items()]
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
