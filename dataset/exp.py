from __future__ import print_function
# pylint: disable-msg=E0611
from time import strftime
from mysql.connector.pooling import MySQLConnectionPool
from avro.io import DatumWriter
from avro.datafile import DataFileWriter
from exporter.wcc_trn import WccTrnExporter
from exporter.kline import KlineExporter
from exporter.wctrain import WcTrainExporter

import avro
import avro.schema
import sys
import os
import multiprocessing
import numpy as np

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

klexp = KlineExporter(cnxpool)
exp_dict = {
    "wcc_trn": WccTrnExporter(cnxpool),
    "kline_d_b": klexp,
    "kline_d_n": klexp,
    "kline_d": klexp,
    "wctrain": WcTrainExporter(cnxpool)
}


def export(table, args):
    dest = args.dest
    print('{} exporting table {}...'.format(
        strftime("%H:%M:%S"), table))
    if exp_dict[table] is None:
        global cnxpool
        cnx = cnxpool.get_connection()
        writer = None
        try:
            query = "SELECT * from {}".format(table)
            cursor = cnx.cursor(dictionary=True)
            cursor.execute(query)
            rows = cursor.fetchall()
            schema = avro.schema.parse(
                open(os.path.join("schema", "{}.avsc".format(table)), "rb").read())
            file_path = os.path.join(dest, "{}.avro".format(table))
            print('{} exporting to {}'.format(
                strftime("%H:%M:%S"), file_path))
            writer = DataFileWriter(
                open(file_path, "wb"), DatumWriter(), schema)
            for i, row in enumerate(rows):
                if i != 0 and i % 5000 == 0:
                    print('{} {} records exported...'.format(
                        strftime("%H:%M:%S"), i))
                writer.append(row)
            cursor.close()
        except:
            print(sys.exc_info()[0])
            raise
        finally:
            cnx.close()
            if writer:
                writer.close()
    else:
        exp_dict[table].export(table, dest, args)
