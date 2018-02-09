from __future__ import print_function
import mysql.connector
import sys
import numpy as np
import tensorflow as tf
from pprint import pprint
from time import strftime


def tagDataTrained(uuids, batch_no):
    cnx = mysql.connector.connect(
        host='localhost', user='mysql', database='secu', password='123456')
    flag = "TRN_{}_{}".format(strftime("%Y%m%d_%H%M%S"), batch_no)
    uuidval = ','.join(["'{}'".format(u) for u in uuids])
    try:
        cursor = cnx.cursor()
        stmt = (
            "update kpts "
            "set flag = '{}' "
            "where uuid in ({})"
        )
        cursor.execute(stmt.format(flag, uuidval))
        cnx.commit()
    except:
        cnx.rollback()
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


def loadTestSet(max_step):
    cnx = mysql.connector.connect(
        host='localhost', user='mysql', database='secu', password='123456')
    try:
        cursor = cnx.cursor(buffered=True)
        pick = (
            "SELECT  "
            "    flag "
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
        query = (
            "SELECT "
            "    SUBSTR(b.date, 1, 4) yyyy, "
            "    SUBSTR(b.date, 6, 2) mm, "
            "    SUBSTR(b.date, 9, 2) dd, "
            "    b.open o, "
            "    b.high h, "
            "    b.low l, "
            "    b.close c, "
            "    n.open n_o, "
            "    n.high n_h, "
            "    n.low n_l, "
            "    n.close n_c, "
            "    p.open p_o, "
            "    p.high p_h, "
            "    p.low p_l, "
            "    p.close p_c, "
            "    p.volume vol, "
            "    p.varate_rgl vr, "
            "    i.kdj_k k, "
            "    i.kdj_d d, "
            "    i.kdj_j j "
            "FROM "
            "    kline_d_b b "
            "        INNER JOIN "
            "    kline_d p USING (code , klid , date) "
            "        INNER JOIN "
            "    kline_d_n n USING (code , klid , date) "
            "        INNER JOIN "
            "    indicator_d i USING (code , klid , date) "
            "WHERE "
            "    code = %s "
            "        AND klid BETWEEN %s AND %s "
            "ORDER BY klid "
            "LIMIT {}"
        )
        data = []   # [batch, max_step, feature]
        labels = []  # [batch, label]  one-hot labels
        uuids = []
        for (uuid, code, klid, score) in cursor:
            uuids.append(uuid)
            label = np.zeros((21), dtype=np.int8)
            label[int(score)+10] = 1
            labels.append(label)
            fcursor = cnx.cursor()
            fcursor.execute(query.format(max_step), (code, klid - 999, klid))
            s_idx = 0
            steps = np.zeros((max_step, 20), dtype='f')
            for (yyyy, mm, dd, o, h, l, c, n_o, n_h, n_l, n_c, p_o, p_h, p_l, p_c, vol, vr, k, d, j) in fcursor:
                steps[s_idx] = [yyyy, mm, dd, o,
                                h, l, c, n_o, n_h, n_l, n_c, p_o, p_h, p_l, p_c, vol, vr, k, d, j]
                s_idx += 1
            data.append(steps)
            fcursor.close()
        cursor.close()
        # pprint(data)
        # print("\n")
        # pprint(len(labels))
        return uuids, data, labels
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


def loadPrepTrainingData(batch_no, max_step):
    cnx = mysql.connector.connect(
        host='localhost', user='mysql', database='secu', password='123456')
    try:
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
        query = (
            "SELECT "
            "    SUBSTR(b.date, 1, 4) yyyy, "
            "    SUBSTR(b.date, 6, 2) mm, "
            "    SUBSTR(b.date, 9, 2) dd, "
            "    b.open o, "
            "    b.high h, "
            "    b.low l, "
            "    b.close c, "
            "    n.open n_o, "
            "    n.high n_h, "
            "    n.low n_l, "
            "    n.close n_c, "
            "    p.open p_o, "
            "    p.high p_h, "
            "    p.low p_l, "
            "    p.close p_c, "
            "    p.volume vol, "
            "    p.varate_rgl vr, "
            "    i.kdj_k k, "
            "    i.kdj_d d, "
            "    i.kdj_j j "
            "FROM "
            "    kline_d_b b "
            "        INNER JOIN "
            "    kline_d p USING (code , klid , date) "
            "        INNER JOIN "
            "    kline_d_n n USING (code , klid , date) "
            "        INNER JOIN "
            "    indicator_d i USING (code , klid , date) "
            "WHERE "
            "    code = %s "
            "        AND klid BETWEEN %s AND %s "
            "ORDER BY klid "
            "LIMIT {}"
        )
        data = []   # [batch, max_step, feature]
        labels = []  # [batch, label]  one-hot labels
        uuids = []
        for (uuid, code, klid, score) in cursor:
            uuids.append(uuid)
            label = np.zeros((21), dtype=np.int8)
            label[int(score)+10] = 1
            labels.append(label)
            fcursor = cnx.cursor()
            fcursor.execute(query.format(max_step), (code, klid - 999, klid))
            s_idx = 0
            steps = np.zeros((max_step, 20), dtype='f')
            for (yyyy, mm, dd, o, h, l, c, n_o, n_h, n_l, n_c, p_o, p_h, p_l, p_c, vol, vr, k, d, j) in fcursor:
                steps[s_idx] = [yyyy, mm, dd, o,
                                h, l, c, n_o, n_h, n_l, n_c, p_o, p_h, p_l, p_c, vol, vr, k, d, j]
                s_idx += 1
            data.append(steps)
            fcursor.close()
        cursor.close()
        # pprint(data)
        # print("\n")
        # pprint(len(labels))
        return uuids, data, labels
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


def loadTrainingData(batch_size, max_step):
    cnx = mysql.connector.connect(
        host='localhost', user='mysql', database='secu', password='123456')
    try:
        cursor = cnx.cursor(buffered=True)
        query = (
            'SELECT '
            '   uuid, code, klid, score '
            'FROM'
            '   kpts '
            'WHERE '
            '   code NOT IN (SELECT'
            '                       code'
            '                  FROM '
            '                       kpts'
            '                 WHERE'
            "                       flag = 'TEST') "
            '   and flag is null '
            'ORDER BY RAND() '
            'LIMIT {}'
        )
        # print(query)
        cursor.execute(query.format(batch_size))
        query = (
            "SELECT "
            "    SUBSTR(b.date, 1, 4) yyyy, "
            "    SUBSTR(b.date, 6, 2) mm, "
            "    SUBSTR(b.date, 9, 2) dd, "
            "    b.open o, "
            "    b.high h, "
            "    b.low l, "
            "    b.close c, "
            "    p.varate_rgl vr, "
            "    i.kdj_k k, "
            "    i.kdj_d d, "
            "    i.kdj_j j "
            "FROM "
            "    kline_d_b b "
            "        INNER JOIN "
            "    kline_d p USING (code , klid , date) "
            "        INNER JOIN "
            "    indicator_d i USING (code , klid , date) "
            "WHERE "
            "    code = %s "
            "        AND klid BETWEEN %s AND %s "
            "ORDER BY klid "
            "LIMIT {}"
        )
        data = []   # [batch, max_step, feature]
        labels = []  # [batch, label]  one-hot labels
        uuids = []
        for (uuid, code, klid, score) in cursor:
            uuids.append(uuid)
            label = np.zeros((21), dtype=np.int8)
            label[int(score)+10] = 1
            labels.append(label)
            fcursor = cnx.cursor()
            fcursor.execute(query.format(max_step), (code, klid - 999, klid))
            s_idx = 0
            steps = np.zeros((max_step, 12), dtype='f')
            for (yyyy, mm, dd, o, h, l, c, vr, k, d, j) in fcursor:
                steps[s_idx] = [yyyy, mm, dd, o, h, l, c, vr, k, d, j]
                s_idx += 1
            data.append(steps)
            fcursor.close()
        cursor.close()
        # pprint(data)
        # print("\n")
        # pprint(len(labels))
        return uuids, data, labels
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()
