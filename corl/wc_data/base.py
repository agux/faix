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


def connect():
    return mysql.connector.connect(
        host='127.0.0.1',
        user='mysql',
        database='secu',
        password='123456',
        # ssl_ca='',
        # use_pure=True,
        connect_timeout=60000)


'''
Loads tagged sample wcc_trn data from database.
Read from backward reinstated klines.
In-place feature-wise standardization is adopted.
'''

k_cols = [
    "lr",
    #  "lr_h", "lr_o", "lr_l", "lr_vol",
]

ftQueryTpl = (
    "SELECT  "
    "    date, "
    "    {0} "  # (d.COLS - mean) / std
    "FROM "
    "    kline_d_b d "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        %s code, "
    "        t.method, "
    "        {1} "  # mean & std fields
    "    FROM "
    "        fs_stats t "
    "    WHERE "
    "        t.method = 'standardization' "
    "    GROUP BY code , t.method) s USING (code) "
    "WHERE "
    "    d.code = %s "
    "    {2} "
    "ORDER BY klid "
    "LIMIT %s "
)

num_cores = multiprocessing.cpu_count()


def getBatch(code, s, e, rcode, max_step, time_shift, ftQueryK, ftQueryD):
    '''
    [max_step, feature*time_shift], length
    '''
    cnx = connect()
    fcursor = cnx.cursor(buffered=True)
    try:
        fcursor.execute(ftQueryK, (code, code, s, e, max_step+time_shift))
        col_names = fcursor.column_names
        featSize = (len(col_names)-1)*2
        total = fcursor.rowcount
        rows = fcursor.fetchall()
        # extract dates and transform to sql 'in' query
        dates = "'{}'".format("','".join([r[0] for r in rows]))
        qd = ftQueryD.format(dates)
        fcursor.execute(qd, (rcode, rcode, max_step+time_shift))
        rtotal = fcursor.rowcount
        r_rows = fcursor.fetchall()
        if total != rtotal:
            raise ValueError(
                "rcode prior data size {} != code's: {}".format(rtotal, total))
        batch = []
        for t in range(time_shift+1):
            steps = np.zeros((max_step, featSize), dtype='f')
            offset = max_step + time_shift - total
            s = max(0, t - offset)
            e = total - time_shift + t
            for i, row in enumerate(rows[s:e]):
                for j, col in enumerate(row[1:]):
                    steps[i+offset][j] = col
            for i, row in enumerate(r_rows[s:e]):
                for j, col in enumerate(row[1:]):
                    steps[i+offset][j+featSize//2] = col
            batch.append(steps)
        return np.concatenate(batch, 1), total - time_shift
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        fcursor.close()
        cnx.close()


def getSeries(uuid, code, klid, rcode, val, max_step, time_shift, ftQueryK, ftQueryD):
    s = max(0, klid-max_step+1-time_shift)
    batch, total = getBatch(
        code, s, klid, rcode, max_step, time_shift, ftQueryK, ftQueryD)
    return uuid, batch, val, total


class DataLoader:
    def __init__(self, time_shift=0, feat_cols=None):
        self.time_shift = time_shift
        self._feat_cols = k_cols if feat_cols is None else feat_cols
        self._qk = None
        self._qd = None

    def loadTestSet(self, max_step, ntest):
        global num_cores
        cnx = connect()
        try:
            rn = np.random.randint(ntest)
            flag = 'TEST_{}'.format(rn)
            print('{} selected test set: {}'.format(
                strftime("%H:%M:%S"), flag))
            query = (
                "SELECT "
                "   uuid, code, klid, rcode, corl "
                "FROM "
                "   wcc_trn "
                "WHERE "
                "   flag = %s "
            )
            cursor = cnx.cursor(buffered=True)
            cursor.execute(query, (flag,))
            tset = cursor.fetchall()
            cursor.close()
            qk, qd = self.getFtQuery()
            r = Parallel(n_jobs=num_cores)(delayed(getSeries)(
                uuid, code, klid, rcode, val, max_step, self.time_shift, qk, qd
            ) for uuid, code, klid, rcode, val in tset)
            uuids, data, vals, seqlen = zip(*r)
            # data = [batch, max_step, feature*time_shift]
            # vals = [batch]
            # seqlen = [batch]
            return np.array(uuids), np.array(data), np.array(vals), np.array(seqlen)
        except:
            print(sys.exc_info()[0])
            raise
        finally:
            cnx.close()

    def loadTrainingData(self, batch_no, max_step):
        global num_cores
        cnx = connect()
        try:
            cursor = cnx.cursor(buffered=True)
            query = (
                'SELECT '
                "   uuid, code, klid, rcode, corl "
                'FROM '
                '   wcc_trn '
                'WHERE '
                "   flag = %s"
            )
            flag = 'TRAIN_{}'.format(batch_no)
            cursor.execute(query, (flag,))
            train_set = cursor.fetchall()
            total = cursor.rowcount
            cursor.close()
            uuids, data, vals, seqlen = [], [], [], []
            if total > 0:
                qk, qd = self.getFtQuery()
                r = Parallel(n_jobs=num_cores)(delayed(getSeries)(
                    uuid, code, klid, rcode, val, max_step, self.time_shift, qk, qd
                ) for uuid, code, klid, rcode, val in train_set)
                uuids, data, vals, seqlen = zip(*r)
            # data = [batch, max_step, feature*time_shift]
            # vals = [batch]
            # seqlen = [batch]
            return np.array(uuids), np.array(data), np.array(vals), np.array(seqlen)
        except:
            print(sys.exc_info()[0])
            raise
        finally:
            cnx.close()

    def getFtQuery(self):
        if self._qk is not None and self._qd is not None:
            return self._qk, self._qd

        k_cols = self._feat_cols
        sep = " "
        p_kline = sep.join(
            ["(d.{0}-s.{0}_mean)/s.{0}_std {0},".format(c) for c in k_cols])
        p_kline = p_kline[:-1]  # strip last comma
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

        self._qk = ftQueryTpl.format(p_kline, stats,
                                     " AND d.klid BETWEEN %s AND %s ")
        self._qd = ftQueryTpl.format(p_kline, stats,
                                     " AND d.date in ({})")
        return self._qk, self._qd
