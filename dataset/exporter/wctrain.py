from __future__ import print_function
# pylint: disable-msg=E0611
from time import strftime
from loky import get_reusable_executor
from mysql.connector.pooling import MySQLConnectionPool

import sys
import os
import multiprocessing
import numpy as np
import errno
import gzip
import json
import ConfigParser

'''
example entry command: 
python main.py --table wctrain --fields lr lr_vol --dest /Users/jx/ProgramData/mysql/dataset 35 4 /Volumes/EXTOS/wc_train
python main.py --table wctrain --fields lr lr_vol --end 55040 --flags TR TS --dest /Volumes/WD-1TB/wcc_train/wc_train 35 4
'''

_executor = None
cnxpool = None
qk, qd = None, None

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


def _getFtQuery(k_cols):
    global qk, qd
    if qk is not None and qd is not None:
        return qk, qd

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

    qk = ftQueryTpl.format(p_kline, stats,
                           " AND d.klid BETWEEN %s AND %s ")
    qd = ftQueryTpl.format(p_kline, stats,
                           " AND d.date in ({})")
    return qk, qd


def _init():
    global cnxpool
    print("PID %d: initializing mysql connection pool..." % os.getpid())
    cnxpool = MySQLConnectionPool(
        pool_name="dbpool",
        pool_size=1,
        host='127.0.0.1',
        port=3306,
        user='mysql',
        database='secu',
        password='123456',
        # ssl_ca='',
        # use_pure=True,
        connect_timeout=60000
    )


def _getExecutor(workers=multiprocessing.cpu_count()):
    global _executor
    if _executor is not None:
        return _executor
    _executor = get_reusable_executor(
        max_workers=workers,
        initializer=_init,
        # initargs=(db_pool_size, db_host, db_port, db_pwd),
        timeout=1800)
    return _executor


def _getBatch(code, s, e, rcode, max_step, time_shift, ftQueryK, ftQueryD):
    '''
    [max_step, feature*time_shift], length
    '''
    global cnxpool
    cnx = cnxpool.get_connection()
    fcursor = cnx.cursor(buffered=True)
    limit = max_step+time_shift
    try:
        fcursor.execute(ftQueryK, (code, code, s, e, limit))
        col_names = fcursor.column_names
        featSize = (len(col_names)-1)*2
        total = fcursor.rowcount
        rows = fcursor.fetchall()
        # extract dates and transform to sql 'in' query
        dates = "'{}'".format("','".join([r[0] for r in rows]))
        qd = ftQueryD.format(dates)
        fcursor.execute(qd, (rcode, rcode, limit))
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


def _getSeries(p):
    # print('{} p: {}'.format(os.getpid(), p))
    code, klid, rcode, val, max_step, time_shift, ftQueryK, ftQueryD = p
    s = max(0, klid-max_step+1-time_shift)
    batch, total = _getBatch(
        code, s, klid, rcode, max_step, time_shift, ftQueryK, ftQueryD)
    return batch, val, total


def _write_file(file_path, payload):
    print('{} exporting {}'.format(
        strftime("%H:%M:%S"), file_path))
    with gzip.GzipFile(file_path, 'wb') as fout:
        fout.write(json.dumps(
            payload, separators=(',', ':')).encode('utf-8'))


def _exp_wctrain(flag, bno, dest, feat_cols, max_step, time_shift, alt_dirs):
    global cnxpool
    vflag = flag
    if flag == 'TR':
        vflag = 'TRAIN'
    elif flag == 'TS':
        vflag = 'TEST'
    file_path = os.path.join(dest, "{}_{}.json.gz".format(vflag, bno))
    tmpf_path = os.path.join(dest, "{}_{}.json.tmp".format(vflag, bno))
    if os.path.exists(file_path):
        print('{} {} {} file already exists, skipping'.format(
            strftime("%H:%M:%S"), flag, bno))
        return os.getpid()
    else:
        if alt_dirs is not None:
            for d in alt_dirs:
                if os.path.exists(file_path):
                    print('{} {} {} file already exists, skipping'.format(
                        strftime("%H:%M:%S"), flag, bno))
                    return os.getpid()

    print('{} loading {}, {}...'.format(strftime("%H:%M:%S"), flag, bno))
    cnx = cnxpool.get_connection()
    try:
        cursor = cnx.cursor(buffered=True)
        cursor.execute(
            "SELECT code, klid, rcode, corl_stz from wcc_trn where flag = %s and bno = %s", (flag, bno,))
        rows = cursor.fetchall()
        cursor.close()
        data, vals, seqlen = [], [], []
        qk, qd = _getFtQuery(feat_cols)
        exc = _getExecutor(2)
        params = [(code, klid, rcode, val, max_step, time_shift, qk, qd)
                  for code, klid, rcode, val in rows]
        r = list(exc.map(_getSeries, params))
        data, vals, seqlen = zip(*r)
        # data = [batch, max_step, feature*time_shift]
        # vals = [batch]
        # seqlen = [batch]
        payload = {
            'features': np.array(data, 'f').tolist(),
            'labels': np.array(vals, 'f').tolist(),
            'seqlens': np.array(seqlen, 'i').tolist()
        }
        _write_file(tmpf_path, payload)
        os.rename(tmpf_path, file_path)
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()
    return os.getpid()


def getMetaInfo():
    global cnxpool
    # get batch_size and count for training/test set
    cnx = cnxpool.get_connection()
    try:
        cursor = cnx.cursor()
        d = {}
        print('{} counting test set...'.format(strftime("%H:%M:%S")))
        cursor.execute(
            "SELECT max(bno) from wcc_trn where flag = 'TS'")
        d['test_count'] = cursor.fetchone()[0]
        print('{} number of test set: {}'.format(
            strftime("%H:%M:%S"), d['test_count']))
        print('{} querying test set batch size...'.format(strftime("%H:%M:%S")))
        cursor.execute(
            "SELECT count(*) from wcc_trn where flag = 'TS' and bno = 1")
        d['test_bsize'] = cursor.fetchone()[0]
        print('{} test set batch size: {}'.format(
            strftime("%H:%M:%S"), d['test_bsize']))

        print('{} counting training set...'.format(strftime("%H:%M:%S")))
        cursor.execute(
            "SELECT max(bno) from wcc_trn where flag = 'TR'")
        d['train_count'] = cursor.fetchone()[0]
        print('{} number of training set: {}'.format(
            strftime("%H:%M:%S"), d['train_count']))
        print('{} querying training set batch size...'.format(strftime("%H:%M:%S")))
        cursor.execute(
            "SELECT count(*) from wcc_trn where flag = 'TR' and bno = 1")
        d['train_bsize'] = cursor.fetchone()[0]
        print('{} training set batch size: {}'.format(
            strftime("%H:%M:%S"), d['train_bsize']))
        return d
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()


class WcTrainExporter:

    def __init__(self, cpool):
        global cnxpool
        cnxpool = cpool

    def export(self, table, dest, args):
        global cnxpool
        feat_cols = args.fields
        start = args.start
        end = args.end
        flags = args.flags
        max_step = int(args.options[0])
        time_shift = int(args.options[1])
        alt_dirs = args.options[2:] if len(args.options) > 2 else None
        assert feat_cols is not None and max_step is not None and time_shift is not None
        print('{} feat_cols: {}'.format(strftime("%H:%M:%S"), feat_cols))
        print('{} max_step: {}'.format(strftime("%H:%M:%S"), max_step))
        print('{} time_shift: {}'.format(strftime("%H:%M:%S"), time_shift))
        cnx = cnxpool.get_connection()
        cursor = None
        try:
            if not os.path.exists(dest):
                try:
                    os.makedirs(dest)
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            print('{} destination: {}'.format(strftime("%H:%M:%S"), dest))
            # export meta file if not exists
            meta_file = os.path.join(dest, 'meta.txt')
            if not os.path.exists(meta_file):
                print('{} writing meta info to {} ...'.format(
                    strftime("%H:%M:%S"), meta_file))
                d = getMetaInfo()
                cfg = ConfigParser.ConfigParser()
                sect_com = 'common'
                cfg.add_section(sect_com)
                cfg.set(sect_com, 'target', 'stock correlation analysis')
                cfg.set(sect_com, 'time_step', str(max_step))
                cfg.set(sect_com, 'time_shift', str(time_shift))
                cfg.set(sect_com, 'features', ', '.join(map(str, feat_cols)))
                feature_size = len(feat_cols) * 2 * (time_shift+1)
                cfg.set(sect_com, 'feature_size', str(feature_size))
                cfg.set(sect_com, 'structure',
                        '[?, {}, {}]'.format(max_step, feature_size))
                sect_test = 'test set'
                cfg.add_section(sect_test)
                cfg.set(sect_test, 'batch_size', str(d['test_bsize']))
                cfg.set(sect_test, 'count', str(d['test_count']))
                sect_train = 'training set'
                cfg.add_section(sect_train)
                cfg.set(sect_train, 'batch_size', str(d['train_bsize']))
                cfg.set(sect_train, 'count', str(d['train_count']))
                with open(meta_file, 'wb+') as f:
                    cfg.write(f)
            print('{} fetching flags...'.format(strftime("%H:%M:%S")))
            query = "SELECT distinct flag, bno from wcc_trn where 1=1"
            if start is not None:
                query += " and bno >= {}".format(start)
            if end is not None:
                query += " and bno <= {}".format(end)
            if flags is not None:
                query += " and flag in ({})".format(
                    ",".join(["'{}'".format(f) for f in flags]))
            query += " order by bno"
            print("{} constructed query: {}".format(strftime("%H:%M:%S"), query))
            cursor = cnx.cursor(buffered=True)
            cursor.execute(query)
            rows = cursor.fetchall()
            total = cursor.rowcount
            cursor.close()
            print('{} num flags: {}'.format(strftime("%H:%M:%S"), total))
            exc = _getExecutor(int(multiprocessing.cpu_count()*0.7))
            for flag, bno in rows:
                exc.submit(_exp_wctrain, flag, bno, dest,
                           feat_cols, max_step, time_shift, alt_dirs)
            exc.shutdown()
        except:
            print(sys.exc_info()[0])
            raise
        finally:
            cnx.close()
