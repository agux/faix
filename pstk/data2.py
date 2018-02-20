from __future__ import print_function
import mysql.connector
import sys
import sqlalchemy as sqla
import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint
from time import strftime

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
    "    lr_vol "
    "FROM "
    "    kline_d "
    "WHERE "
    "    code = %s "
    "        AND klid BETWEEN %s AND %s "
    "ORDER BY klid "
    "LIMIT {}"
)


def loadTestSet(max_step):
    cnx = mysql.connector.connect(
        host='localhost', user='mysql', database='secu', password='123456')
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
        data = []   # [batch, max_step, feature]
        labels = []  # [batch, label]  one-hot labels
        uuids = []
        for (uuid, code, klid, score) in cursor:
            uuids.append(uuid)
            label = np.zeros(nclass, dtype=np.int8)
            label[int(score)+nclass//2] = 1
            labels.append(label)
            fcursor = cnx.cursor()
            fcursor.execute(ftQuery.format(max_step),
                            (code, klid - max_step+1, klid))
            s_idx = 0
            steps = np.zeros((max_step, len(fcursor.column_names)), dtype='f')
            for row in fcursor:
                steps[s_idx] = [col for col in row]
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


def loadTrainingData(batch_no, max_step):
    cnx = mysql.connector.connect(
        host='localhost', user='mysql', database='secu', password='123456')
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
        data = []   # [batch, max_step, feature]
        labels = []  # [batch, label]  one-hot labels
        uuids = []
        for (uuid, code, klid, score) in cursor:
            uuids.append(uuid)
            label = np.zeros((nclass), dtype=np.int8)
            label[int(score)+nclass//2] = 1
            labels.append(label)
            fcursor = cnx.cursor()
            fcursor.execute(ftQuery.format(max_step),
                            (code, klid - max_step+1, klid))
            s_idx = 0
            steps = np.zeros((max_step, len(fcursor.column_names)), dtype='f')
            for row in fcursor:
                steps[s_idx] = [col for col in row]
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
