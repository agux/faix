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
Features: OHLCVAX-LR + MA-LR(8*2) + SH & SZ indices-LR((6+8*2)*2) (7+16+44=67)

Label format: Scalar
'''

TIME_SHIFT = 9

nclsQry = (
    "SELECT  "
    "    COUNT(*) "
    "FROM "
    "    (SELECT DISTINCT "
    "        score "
    "    FROM "
    "        kpts) t"
)

ftQuery = (
    "SELECT  "
    "    d.lr, "
    "    d.lr_h, "
    "    d.lr_o, "
    "    d.lr_l, "
    "    d.lr_vol, "
    "    d.lr_amt, "
    "    d.lr_xr, "
    "    d.lr_ma5, "
    "    d.lr_ma10, "
    "    d.lr_ma20, "
    "    d.lr_ma30, "
    "    d.lr_ma60, "
    "    d.lr_ma120, "
    "    d.lr_ma200, "
    "    d.lr_ma250, "
    "    d.lr_vol5, "
    "    d.lr_vol10, "
    "    d.lr_vol20, "
    "    d.lr_vol30, "
    "    d.lr_vol60, "
    "    d.lr_vol120, "
    "    d.lr_vol200, "
    "    d.lr_vol250, "
    "    COALESCE(sh.lr,0) sh_lr, "
    "    COALESCE(sh.lr_h,0) sh_lr_h, "
    "    COALESCE(sh.lr_o,0) sh_lr_o, "
    "    COALESCE(sh.lr_l,0) sh_lr_l, "
    "    COALESCE(sh.lr_vol,0) sh_lr_vol, "
    "    COALESCE(sh.lr_amt,0) sh_lr_amt, "
    "    COALESCE(sh.lr_ma5,0) sh_lr_ma5, "
    "    COALESCE(sh.lr_ma10,0) sh_lr_ma10, "
    "    COALESCE(sh.lr_ma20,0) sh_lr_ma20, "
    "    COALESCE(sh.lr_ma30,0) sh_lr_ma30, "
    "    COALESCE(sh.lr_ma60,0) sh_lr_ma60, "
    "    COALESCE(sh.lr_ma120,0) sh_lr_ma120, "
    "    COALESCE(sh.lr_ma200,0) sh_lr_ma200, "
    "    COALESCE(sh.lr_ma250,0) sh_lr_ma250, "
    "    COALESCE(sh.lr_vol5,0) sh_lr_vol5, "
    "    COALESCE(sh.lr_vol10,0) sh_lr_vol10, "
    "    COALESCE(sh.lr_vol20,0) sh_lr_vol20, "
    "    COALESCE(sh.lr_vol30,0) sh_lr_vol30, "
    "    COALESCE(sh.lr_vol60,0) sh_lr_vol60, "
    "    COALESCE(sh.lr_vol120,0) sh_lr_vol120, "
    "    COALESCE(sh.lr_vol200,0) sh_lr_vol200, "
    "    COALESCE(sh.lr_vol250,0) sh_lr_vol250, "
    "    COALESCE(sz.lr,0) sz_lr, "
    "    COALESCE(sz.lr_h,0) sz_h, "
    "    COALESCE(sz.lr_o,0) sz_o, "
    "    COALESCE(sz.lr_l,0) sz_l, "
    "    COALESCE(sz.lr_vol,0) sz_vol, "
    "    COALESCE(sz.lr_amt,0) sz_lr_amt, "
    "    COALESCE(sz.lr_ma5,0) sz_lr_ma5, "
    "    COALESCE(sz.lr_ma10,0) sz_lr_ma10, "
    "    COALESCE(sz.lr_ma20,0) sz_lr_ma20, "
    "    COALESCE(sz.lr_ma30,0) sz_lr_ma30, "
    "    COALESCE(sz.lr_ma60,0) sz_lr_ma60, "
    "    COALESCE(sz.lr_ma120,0) sz_lr_ma120, "
    "    COALESCE(sz.lr_ma200,0) sz_lr_ma200, "
    "    COALESCE(sz.lr_ma250,0) sz_lr_ma250, "
    "    COALESCE(sz.lr_vol5,0) sz_lr_vol5, "
    "    COALESCE(sz.lr_vol10,0) sz_lr_vol10, "
    "    COALESCE(sz.lr_vol20,0) sz_lr_vol20, "
    "    COALESCE(sz.lr_vol30,0) sz_lr_vol30, "
    "    COALESCE(sz.lr_vol60,0) sz_lr_vol60, "
    "    COALESCE(sz.lr_vol120,0) sz_lr_vol120, "
    "    COALESCE(sz.lr_vol200,0) sz_lr_vol200, "
    "    COALESCE(sz.lr_vol250,0) sz_lr_vol250 "
    "FROM "
    "    kline_d d "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        lr, lr_h, lr_o, lr_l, lr_vol, lr_amt, "
    "        lr_ma5, lr_ma10, lr_ma20, lr_ma30, lr_ma60, lr_ma120, lr_ma200, lr_ma250, "
    "        lr_vol5, lr_vol10, lr_vol20, lr_vol30, lr_vol60, lr_vol120, lr_vol200, lr_vol250, "
    "        date "
    "    FROM "
    "        kline_d "
    "    WHERE "
    "        code = 'sh000001') sh USING (date) "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        lr, lr_h, lr_o, lr_l, lr_vol, lr_amt, "
    "        lr_ma5, lr_ma10, lr_ma20, lr_ma30, lr_ma60, lr_ma120, lr_ma200, lr_ma250, "
    "        lr_vol5, lr_vol10, lr_vol20, lr_vol30, lr_vol60, lr_vol120, lr_vol200, lr_vol250, "
    "        date "
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
            "    kpts "
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
            "   kpts "
            "WHERE "
            "   flag = '{}' "
        )
        cursor.execute(query.format(row[0]))
        kpts = cursor.fetchall()
        cursor.close()
        data = []   # [batch, max_step, feature*time_shift]
        labels = []  # [batch]  scalar labels
        seqlen = []  # [batch]  scalar sequence length
        uuids = []
        for (uuid, code, klid, score) in kpts:
            uuids.append(uuid)
            labels.append(score)
            s = max(0, klid-max_step+1-TIME_SHIFT)
            batch, total = getBatch(cnx, code, s, klid, max_step)
            data.append(batch)
            seqlen.append(total)
        return uuids, np.array(data), np.array(labels), np.array(seqlen), nclass
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


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
        for t in range(TIME_SHIFT+1):
            steps = np.zeros((max_step, featSize), dtype='f')
            offset = max_step + TIME_SHIFT - total
            s = max(0, t - offset)
            e = total - TIME_SHIFT + t
            for i, row in enumerate(rows[s:e]):
                steps[i+offset] = [col for col in row]
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
            '   kpts '
            'WHERE '
            "   flag = 'TRN_{}'"
        )
        # print(query)
        cursor.execute(query.format(batch_no))
        kpts = cursor.fetchall()
        cursor.close()
        data = []   # [batch, max_step, feature*time_shift]
        labels = []  # [batch]  scalar labels
        seqlen = []  # [batch]  scalar sequence lengths
        uuids = []
        for (uuid, code, klid, score) in kpts:
            uuids.append(uuid)
            labels.append(score)
            s = max(0, klid-max_step+1-TIME_SHIFT)
            batch, total = getBatch(cnx, code, s, klid, max_step)
            data.append(batch)
            seqlen.append(total)
        # pprint(data)
        # print("\n")
        # pprint(len(labels))
        return uuids, np.array(data), np.array(labels), np.array(seqlen), nclass
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()
