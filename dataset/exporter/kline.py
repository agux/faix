from __future__ import print_function
# pylint: disable-msg=E0611
from time import strftime
from loky import get_reusable_executor
from mysql.connector.pooling import MySQLConnectionPool
from avro.io import DatumWriter
from avro.datafile import DataFileWriter

import avro
import avro.schema
import sys
import os
import multiprocessing
import numpy as np
import errno

exp_fields = ['code', 'date', 'klid', 'lr', 'lr_h', 'lr_h_c', 'lr_o', 'lr_o_c',
              'lr_l', 'lr_l_c', 'lr_vol', 'lr_ma5', 'lr_ma5_o', 'lr_ma5_h', 'lr_ma5_l',
              'lr_ma10', 'lr_ma10_o', 'lr_ma10_h', 'lr_ma10_l', 'lr_ma20', 'lr_ma20_o',
              'lr_ma20_h', 'lr_ma20_l', 'lr_ma30', 'lr_ma30_o', 'lr_ma30_h', 'lr_ma30_l',
              'lr_ma60', 'lr_ma60_o', 'lr_ma60_h', 'lr_ma60_l', 'lr_ma120', 'lr_ma120_o',
              'lr_ma120_h', 'lr_ma120_l', 'lr_ma200', 'lr_ma200_o', 'lr_ma200_h', 'lr_ma200_l',
              'lr_ma250', 'lr_ma250_o', 'lr_ma250_h', 'lr_ma250_l', 'lr_vol5', 'lr_vol10',
              'lr_vol20', 'lr_vol30', 'lr_vol60', 'lr_vol120', 'lr_vol200', 'lr_vol250',
              'udate', 'utime']

field_str = ','.join(exp_fields)

parallel_threshold = 2 ** 21

_executor = None
cnxpool = None
file_path = None
count = 0
schema = None


def _init():
    global cnxpool, schema
    schema = avro.schema.parse(
        open(os.path.join("schema", "kline.avsc"), "rb").read())
    print("PID %d: initializing mysql connection pool..." % os.getpid())
    cnxpool = MySQLConnectionPool(
        pool_name="dbpool",
        pool_size=3,
        host='127.0.0.1',
        port=3306,
        user='mysql',
        database='secu',
        password='123456',
        # ssl_ca='',
        # use_pure=True,
        connect_timeout=60000
    )


def _getExecutor():
    global _executor
    if _executor is not None:
        return _executor
    _executor = get_reusable_executor(
        max_workers=multiprocessing.cpu_count(),
        initializer=_init,
        # initargs=(db_pool_size, db_host, db_port, db_pwd),
        timeout=1800)
    return _executor


def _exp_kline(p):
    global cnxpool, count, file_path, schema, field_str
    table, code, dest = p
    print('{} [{}] exporting {}...'.format(
        strftime("%H:%M:%S"), os.getpid(), code))
    cnx = cnxpool.get_connection()
    writer = None
    _schema = None
    if file_path is None or count >= parallel_threshold:
        file_path = os.path.join(
            dest, table, "{}_{}.avro".format(os.getpid(), strftime("%Y%m%d_%H%M%S")))
        print('{} allocating new file {}...'.format(
            strftime("%H:%M:%S"), file_path))
        count = 0
        _schema = schema
    try:
        cursor = cnx.cursor(dictionary=True, buffered=True)
        cursor.execute("SELECT {} from {} where code = %s".format(
            field_str, table), (code,))
        rows = cursor.fetchall()
        total = cursor.rowcount
        cursor.close()
        writer = DataFileWriter(
            open(file_path, "ab+"), DatumWriter(), _schema)
        for row in rows:
            writer.append(row)
        count += total
    except:
        print(sys.exc_info()[0])
        raise
    finally:
        cnx.close()
        if writer:
            writer.close()


class KlineExporter:

    def __init__(self, cpool):
        self._cpool = cpool

    def export(self, table, dest, args):
        cnx = self._cpool.get_connection()
        cursor = None
        try:
            file_dir = os.path.join(dest, table)
            if not os.path.exists(file_dir):
                try:
                    os.makedirs(file_dir)
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            print('{} fetching security codes from {}...'.format(
                strftime("%H:%M:%S"), table))
            query = "SELECT distinct code from {}".format(table)
            cursor = cnx.cursor(buffered=True)
            cursor.execute(query)
            rows = cursor.fetchall()
            total = cursor.rowcount
            cursor.close()
            print('{} num codes: {}'.format(strftime("%H:%M:%S"), total))
            params = [(table, row[0], dest) for row in rows]
            exc = _getExecutor()
            exc.map(_exp_kline, params)
        except:
            print(sys.exc_info()[0])
            raise
        finally:
            cnx.close()
