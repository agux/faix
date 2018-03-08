from __future__ import print_function
import mysql.connector
import sys
import sqlalchemy as sqla
import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint
from time import strftime


def connect():
    return mysql.connector.connect(
        host='localhost', user='mysql', database='secu', password='123456')

ftQuery = (
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
    "    i.kdj_j j, "
    "    COALESCE(shidx.sh_o, 0) sh_o, "
    "    COALESCE(shidx.sh_h, 0) sh_h, "
    "    COALESCE(shidx.sh_c, 0) sh_c, "
    "    COALESCE(shidx.sh_l, 0) sh_l, "
    "    COALESCE(shidx.sh_vol, 0) sh_vol, "
    "    COALESCE(szidx.sz_o, 0) sz_o, "
    "    COALESCE(szidx.sz_h, 0) sz_h, "
    "    COALESCE(szidx.sz_c, 0) sz_c, "
    "    COALESCE(szidx.sz_l, 0) sz_l, "
    "    COALESCE(szidx.sz_vol, 0) sz_vol "
    "FROM "
    "    kline_d_b b "
    "        INNER JOIN "
    "    kline_d p USING (code , klid , date) "
    "        INNER JOIN "
    "    kline_d_n n USING (code , klid , date) "
    "        INNER JOIN "
    "    indicator_d i USING (code , klid , date) "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        date, "
    "            open sh_o, "
    "            high sh_h, "
    "            close sh_c, "
    "            low sh_l, "
    "            volume sh_vol "
    "    FROM "
    "        kline_d "
    "    WHERE "
    "        code = 'sh000001') shidx USING (date) "
    "        LEFT OUTER JOIN "
    "    (SELECT  "
    "        date, "
    "            open sz_o, "
    "            high sz_h, "
    "            close sz_c, "
    "            low sz_l, "
    "            volume sz_vol "
    "    FROM "
    "        kline_d "
    "    WHERE "
    "        code = 'sz399001') szidx USING (date) "
    "WHERE "
    "    code = %s "
    "        AND klid BETWEEN %s AND %s "
    "ORDER BY klid "
    "LIMIT {}"
)


def tagDataTrained(uuids, batch_no):
    cnx = connect()
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
    cnx = connect()
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
        data = []   # [batch, max_step, feature]
        labels = []  # [batch, label]  one-hot labels
        uuids = []
        for (uuid, code, klid, score) in cursor:
            uuids.append(uuid)
            label = np.zeros((21), dtype=np.int8)
            label[int(score)+10] = 1
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


def loadPrepTrainingData(batch_no, max_step):
    cnx = connect()
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
        data = []   # [batch, max_step, feature]
        labels = []  # [batch, label]  one-hot labels
        uuids = []
        for (uuid, code, klid, score) in cursor:
            uuids.append(uuid)
            label = np.zeros((21), dtype=np.int8)
            label[int(score)+10] = 1
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


def loadPrepTrainingData4D(batch_no, max_step):
    '''
    returned format: [batch_size, max_step, interleaved_dim, feature_size]
    '''
    cnx = connect()
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
        numRels = getNumRelatives(cnx)
        rels = getRelativeData()
        relCodes = rels.index.get_level_values('code').tolist()
        cursor.execute(query.format(batch_no))
        data = []   # [batch, max_step, feature]
        labels = []  # [batch, label]  one-hot labels
        uuids = []
        ftQuery = (
            "SELECT "
            "    b.date d, "
            "    b.open o, "
            "    b.high h, "
            "    b.low l, "
            "    b.close c, "
            "    b.volume v "
            "FROM "
            "    kline_d_b b "
            "WHERE "
            "    b.code = %s "
            "        AND b.klid BETWEEN %s AND %s "
            "ORDER BY b.klid "
            "LIMIT {}"
        )
        for (uuid, code, klid, score) in cursor:
            uuids.append(uuid)
            label = np.zeros((21), dtype=np.int8)
            label[int(score)+10] = 1
            labels.append(label)
            fcursor = cnx.cursor()
            fcursor.execute(ftQuery.format(max_step),
                            (code, klid - max_step+1, klid))
            s_idx = 0
            steps = np.zeros(
                (max_step, numRels*2, len(fcursor.column_names)-1))
            for row in fcursor:
                print("#{} populating {} {}".format(s_idx, code, row[0]))
                for r in range(numRels*2):
                    if r % 2 == 0:
                        steps[s_idx][r] = [col for i,
                                           col in enumerate(row) if i > 0]
                    else:
                        try:
                            steps[s_idx][r] = rels.loc[relCodes[r//2]
                                                       ].loc[row[0]].as_matrix()
                        except KeyError:
                            # not found
                            pass
                        except Exception:
                            raise
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


relData = None


def getRelativeData():
    global relData
    if relData is not None:
        return relData
    print("fetching relative data from database, may take a while...")
    sql = (
        "SELECT "
        "    d.code, d.date, d.open, d.high, d.low, d.close, d.volume "
        "FROM "
        "    kline_d d "
        "        INNER JOIN "
        "    idxlst USING (code)  "
        "UNION ALL SELECT  "
        "    * "
        "FROM "
        "    (SELECT  "
        "        b.code, b.date, b.open, b.high, b.close, b.low, b.volume "
        "    FROM "
        "        kline_d_b b "
        "    INNER JOIN cmpool p USING (code) "
        "    ORDER BY p.seqno) t"
    )
    relData = pd.read_sql_query(sql,
                                sqla.create_engine(
                                    'mysql+mysqlconnector://mysql:123456@localhost/secu'),
                                index_col=["code", "date"])
    print("relative data fetched from database: {}".format(relData.shape))
    return relData


def getNumRelatives(conn):
    c = conn.cursor()
    c.execute("select count(*) cnt from cmpool")
    n = np.asarray(c.fetchone())[0]
    c.close()
    return n


def loadTrainingData(batch_size, max_step):
    cnx = connect()
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
            fcursor.execute(query.format(max_step),
                            (code, klid - max_step+1, klid))
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
