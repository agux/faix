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

parallel_threshold = 2 ** 21

_executor = None
cnxpool = None
file_path = None
count = 0
schema = None


def _init():
    global cnxpool, schema
    schema = avro.schema.parse(
        open(os.path.join("schema", "wcc_trn.avsc"), "rb").read())
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


def _exp_wcctrn(p):
    global cnxpool, count, file_path, schema
    flag, dest = p
    print('{} [{}] exporting {}...'.format(
        strftime("%H:%M:%S"), os.getpid(), flag))
    cnx = cnxpool.get_connection()
    writer = None
    _schema = None
    if file_path is None or count >= parallel_threshold:
        file_path = os.path.join(
            dest, "wcc_trn", "{}_{}.avro".format(os.getpid(), strftime("%Y%m%d_%H%M%S")))
        print('{} allocating new file {}...'.format(
            strftime("%H:%M:%S"), file_path))
        count = 0
        _schema = schema
    try:
        cursor = cnx.cursor(dictionary=True, buffered=True)
        cursor.execute("SELECT * from wcc_trn where flag = %s", (flag,))
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


class WccTrnExporter:

    def __init__(self, cpool):
        self._cpool = cpool

    def export(self, dest):
        cnx = self._cpool.get_connection()
        cursor = None
        try:
            file_dir = os.path.join(dest, "wcc_trn")
            if not os.path.exists(file_dir):
                try:
                    os.makedirs(file_dir)
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            print('{} fetching flags...'.format(strftime("%H:%M:%S")))
            query = "SELECT distinct flag from wcc_trn"
            cursor = cnx.cursor(buffered=True)
            cursor.execute(query)
            rows = cursor.fetchall()
            total = cursor.rowcount
            cursor.close()
            print('{} num flags: {}'.format(strftime("%H:%M:%S"), total))
            params = [(row[0], dest) for row in rows]
            exc = _getExecutor()
            exc.map(_exp_wcctrn, params)
        except:
            print(sys.exc_info()[0])
            raise
        finally:
            cnx.close()
