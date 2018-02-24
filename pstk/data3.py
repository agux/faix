from __future__ import print_function
import mysql.connector
import sys
import sqlalchemy as sqla
import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint
from time import strftime

TIME_SHIFT = 3

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
    "SELECT "
    "    lr, "
    "    lr_h,"
    "    lr_o,"
    "    lr_l,"
    "    lr_vol "
    "FROM "
    "    kline_d "
    "WHERE "
    "    code = %s "
    "        AND klid BETWEEN %s AND %s "
    "ORDER BY klid "
    "LIMIT %s "
)

def connect():
    return mysql.connector.connect(
        host='localhost', user='mysql', database='secu', password='123456')


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
        data = []   # [time_shift, batch, max_step, feature]
        labels = []  # [batch, label]  one-hot labels
        uuids = []
        for ts in range(TIME_SHIFT-1, -1, -1):
            batch = []  # [batch, max_step, feature]
            for (uuid, code, klid, score) in kpts:
                if ts == TIME_SHIFT-1:
                    uuids.append(uuid)
                    label = np.zeros(nclass, dtype=np.int8)
                    label[int(score)+nclass//2] = 1
                    labels.append(label)
                s = klid-max_step+1-ts
                e = klid-ts
                batch.append(getBatch(cnx, code, s, e, max_step))
            data.append(batch)
        return uuids, np.array(data), np.array(labels)
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


def getBatch(cnx, code, s, e, max_step):
    fcursor = cnx.cursor(buffered=True)
    try:
        fcursor.execute(ftQuery, (code, s, e, max_step))
        steps = np.zeros(
            (max_step, len(fcursor.column_names)), dtype='f')
        for i, row in enumerate(fcursor):
            steps[i] = [col for col in row]
        return steps
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
        data = []   # [time_shift, batch, max_step, feature]
        labels = []  # [batch, label]  one-hot labels
        uuids = []
        for ts in range(TIME_SHIFT-1, -1, -1):
            batch = []  # [batch, max_step, feature]
            for (uuid, code, klid, score) in kpts:
                if ts == TIME_SHIFT-1:
                    uuids.append(uuid)
                    label = np.zeros((nclass), dtype=np.int8)
                    label[int(score)+nclass//2] = 1
                    labels.append(label)
                s = klid-max_step+1-ts
                e = klid-ts
                batch.append(getBatch(cnx, code, s, e, max_step))
            data.append(batch)
        # pprint(data)
        # print("\n")
        # pprint(len(labels))
        return uuids, np.array(data), np.array(labels)
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()
