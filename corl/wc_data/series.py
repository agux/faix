# from __future__ import absolute_import, division, print_function, unicode_literals

from time import strftime
from mysql.connector.pooling import MySQLConnectionPool
import sys
import ray
import numpy as np

cnxpool = None


def _init(db_pool_size=None, db_host=None, db_port=None, db_pwd=None):
    global cnxpool
    size = db_pool_size or 5
    print("{} initializing mysql connection pool of size {}...".format(
        strftime("%H:%M:%S"), size))
    cnxpool = MySQLConnectionPool(
        pool_name="dbpool",
        pool_size=size,
        host=db_host or '127.0.0.1',
        port=db_port or 3306,
        user='mysql',
        database='secu',
        password=db_pwd or '123456',
        # ssl_ca='',
        # use_pure=True,
        connect_timeout=90000)
    return cnxpool


# @ray.remote
def getSeries(code, klid, rcode, val, shared_args):
    # code, klid, rcode, val, max_step, time_shift, qk, qd = p
    max_step = shared_args['max_step']
    time_shift = shared_args['time_shift']
    s = max(0, klid - max_step + 1 - time_shift)
    batch, total = _getBatch(code, s, klid, rcode, shared_args)
    return batch, val, total


@ray.remote
class DataLoader(object):
    def __init__(self, shared_args):
        host = shared_args['db_host']
        port = shared_args['db_port']
        pwd = shared_args['db_pwd']
        self.conn_pool = _init(1, host, port, pwd)
        self.max_step = shared_args['max_step']
        self.time_shift = shared_args['time_shift']
        self.corl_prior = shared_args['corl_prior']
        self.offset = self.max_step+self.time_shift-1
        self.qk = shared_args['qk2']
        self.qd_idx = shared_args['qd_idx']
        self.index_list = shared_args['index_list']
        self.qd = shared_args['qd']
        self.sqlt_wr_part = (
            "SELECT "
            "    t.code, t_pre.date start_date, t.date end_date, t.klid "
            "FROM "
            "    kline_d_b_lr t_pre, "
            "    (SELECT  "
            "        code, date, klid "
            "    FROM "
            "        kline_d_b_lr PARTITION (%s) "
            "    WHERE "
            "        klid >= %s "
            "            AND (code , date) NOT IN (SELECT "
            "                code, date "
            "            FROM "
            "                wcc_predict) "
            "            {cond} "
            "    ) t "
            "WHERE "
            "    t_pre.code = t.code "
            "        AND t_pre.klid = t.klid - %s "
        )

    def get_wcc_infer_work_request(self, part, cond):
        '''
        Returns code, start_date, end_date, klid for the given table partition.
        '''
        part = part[0]
        cnxpool = self.conn_pool
        cnx = cnxpool.get_connection()
        fcursor = None
        try:
            print('{} querying workload for partition {}'.format(
                strftime("%H:%M:%S"), part))
            fcursor = cnx.cursor(buffered=True)
            fcursor.execute(self.sqlt_wr_part.format(cond=cond),
                            (part, self.corl_prior, self.offset))
            rows = fcursor.fetchall()
            total = fcursor.rowcount
            print('{} workload for partition {}: {}'.format(
                strftime("%H:%M:%S"), part, total))
        except:
            print(sys.exc_info()[0])
            raise
        finally:
            if fcursor is not None:
                fcursor.close()
            cnx.close()
        return [(c, d1, d2, k) for c, d1, d2, k in rows]

    def get_series(self, code, klid, date_start, date_end, rcode, val):
        s = max(0, klid - self.max_step + 1 - self.time_shift)

        batch = self._get_batch(code, s, klid, date_start,
                                date_end, rcode)

        if val is not None:
            return batch, val
        else:
            return batch

    def _get_batch(self, code, s, e, date_start, date_end, rcode):
        cnxpool = self.conn_pool
        max_step = self.max_step
        time_shift = self.time_shift
        qk = self.qk
        qd = self.qd_idx if rcode in self.index_list else self.qd
        cnx = cnxpool.get_connection()
        fcursor = None
        try:
            fcursor = cnx.cursor(buffered=True)
            ymStr = self._get_ymstr(date_start, date_end)
            qk = qk.format(ymStr)
            fcursor.execute(qk, (code, s, e, max_step + time_shift))
            col_names = fcursor.column_names
            featSize = (len(col_names) - 1) * 2
            total = fcursor.rowcount
            rows = fcursor.fetchall()
            # extract dates and transform to sql 'in' query
            dates = [r[0] for r in rows]
            dateStr = "'{}'".format("','".join(dates))
            ym = {d.replace('-', '')[:-2]
                  for d in dates}  # extract year-month to a set
            ymStr = ",".join(ym)
            qd = qd.format(ymStr, dateStr)
            fcursor.execute(qd, (rcode, max_step + time_shift))
            rtotal = fcursor.rowcount
            r_rows = fcursor.fetchall()
            if total != rtotal:
                raise ValueError("{} prior data size {} != {}'s: {}, max_step: {}, time_shift: {}, query: {}".format(
                    rcode, rtotal, code, total, max_step, time_shift, qd))
            batch = []
            for t in range(time_shift + 1):
                steps = np.zeros((max_step, featSize), dtype=np.float32)
                offset = max_step + time_shift - total
                s = max(0, t - offset)
                e = total - time_shift + t
                for i, row in enumerate(rows[s:e]):
                    for j, col in enumerate(row[1:]):
                        steps[i + offset][j] = col
                for i, row in enumerate(r_rows[s:e]):
                    for j, col in enumerate(row[1:]):
                        steps[i + offset][j + featSize // 2] = col
                batch.append(steps)
            return np.concatenate(batch, 1).astype(np.float32)
        except:
            print(sys.exc_info()[0])
            raise
        finally:
            if fcursor is not None:
                fcursor.close()
            cnx.close()

    def _get_ymstr(self, date_start, date_end):
        sy, sm, _ = date_start.split('-')
        ey, em, _ = date_end.split('-')
        sy, sm, ey, em = int(sy), int(sm), int(ey), int(em)
        ymStr = '{:d}{:02d}'.format(sy, sm)
        while not (sy == ey and sm == em):
            sm += 1
            if sm > 12:
                sm = 1
                sy += 1
            ymStr += ',{:d}{:02d}'.format(sy, sm)
        return ymStr


@ray.remote
def getSeries_v2(code, klid, rcode, val, shared_args):
    # code, klid, rcode, val, max_step, time_shift, qk, qd = p
    max_step = shared_args['max_step']
    time_shift = shared_args['time_shift']
    s = max(0, klid - max_step + 1 - time_shift)
    batch = _getBatch_v2(code, s, klid, rcode, shared_args)
    if val is not None:
        return batch, val
    else:
        return batch


def _getBatch(code, s, e, rcode, shared_args):
    '''
    Returns:
    [max_step, feature*time_shift], length
    '''
    global cnxpool
    if cnxpool is None:
        db_host = shared_args['db_host']
        db_port = shared_args['db_port']
        db_pwd = shared_args['db_pwd']
        _init(1, db_host, db_port, db_pwd)

    max_step = shared_args['max_step']
    time_shift = shared_args['time_shift']
    qk = shared_args['qk']
    qd = shared_args['qd_idx'] if rcode in shared_args[
        'index_list'] else shared_args['qd']
    cnx = cnxpool.get_connection()
    fcursor = cnx.cursor(buffered=True)
    try:
        fcursor.execute(qk, (code, s, e, max_step + time_shift))
        col_names = fcursor.column_names
        featSize = (len(col_names) - 1) * 2
        total = fcursor.rowcount
        rows = fcursor.fetchall()
        # extract dates and transform to sql 'in' query
        dates = [r[0] for r in rows]
        dateStr = "'{}'".format("','".join(dates))
        ym = {d.replace('-', '')[:-2]
              for d in dates}  # extract year-month to a set
        ymStr = ",".join(ym)
        qd = qd.format(ymStr, dateStr)
        fcursor.execute(qd, (rcode, max_step + time_shift))
        rtotal = fcursor.rowcount
        r_rows = fcursor.fetchall()
        if total != rtotal:
            raise ValueError("{} prior data size {} != {}'s: {}, max_step: {}, time_shift: {}, query: {}".format(
                rcode, rtotal, code, total, max_step, time_shift, qd))
        batch = []
        for t in range(time_shift + 1):
            steps = np.zeros((max_step, featSize), dtype=np.float32)
            offset = max_step + time_shift - total
            s = max(0, t - offset)
            e = total - time_shift + t
            for i, row in enumerate(rows[s:e]):
                for j, col in enumerate(row[1:]):
                    steps[i + offset][j] = col
            for i, row in enumerate(r_rows[s:e]):
                for j, col in enumerate(row[1:]):
                    steps[i + offset][j + featSize // 2] = col
            batch.append(steps)
        return np.concatenate(batch, 1), total - time_shift
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        fcursor.close()
        cnx.close()


def _getBatch_v2(code, s, e, rcode, shared_args, pool=None):
    '''
    Returns:
    [max_step, feature*time_shift]
    '''
    global cnxpool
    if cnxpool is None:
        if not pool is None:
            cnxpool = pool
        else:
            db_host = shared_args['db_host']
            db_port = shared_args['db_port']
            db_pwd = shared_args['db_pwd']
            _init(1, db_host, db_port, db_pwd)

    max_step = shared_args['max_step']
    time_shift = shared_args['time_shift']
    qk = shared_args['qk']
    qd = shared_args['qd_idx'] if rcode in shared_args[
        'index_list'] else shared_args['qd']
    cnx = cnxpool.get_connection()
    fcursor = None
    try:
        fcursor = cnx.cursor(buffered=True)
        fcursor.execute(qk, (code, s, e, max_step + time_shift))
        col_names = fcursor.column_names
        featSize = (len(col_names) - 1) * 2
        total = fcursor.rowcount
        rows = fcursor.fetchall()
        # extract dates and transform to sql 'in' query
        dates = [r[0] for r in rows]
        dateStr = "'{}'".format("','".join(dates))
        ym = {d.replace('-', '')[:-2]
              for d in dates}  # extract year-month to a set
        ymStr = ",".join(ym)
        qd = qd.format(ymStr, dateStr)
        fcursor.execute(qd, (rcode, max_step + time_shift))
        rtotal = fcursor.rowcount
        r_rows = fcursor.fetchall()
        if total != rtotal:
            raise ValueError("{} prior data size {} != {}'s: {}, max_step: {}, time_shift: {}, query: {}".format(
                rcode, rtotal, code, total, max_step, time_shift, qd))
        batch = []
        for t in range(time_shift + 1):
            steps = np.zeros((max_step, featSize), dtype=np.float32)
            offset = max_step + time_shift - total
            s = max(0, t - offset)
            e = total - time_shift + t
            for i, row in enumerate(rows[s:e]):
                for j, col in enumerate(row[1:]):
                    steps[i + offset][j] = col
            for i, row in enumerate(r_rows[s:e]):
                for j, col in enumerate(row[1:]):
                    steps[i + offset][j + featSize // 2] = col
            batch.append(steps)
        return np.concatenate(batch, 1).astype(np.float32)
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        if fcursor is not None:
            fcursor.close()
        cnx.close()
