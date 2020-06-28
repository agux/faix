import pandas as pd
import numpy as np
from mysql.connector.pooling import MySQLConnectionPool

cnxpool = None

def _parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_host',
                        type=str,
                        help='database host address',
                        default=None)
    parser.add_argument('--db_port',
                        type=int,
                        help='database listening port',
                        default=None)
    parser.add_argument('--db_pwd',
                        type=str,
                        help='database password',
                        default=None)
    return parser.parse_args()

def _init(db_pool_size=None, db_host=None, db_port=None, db_pwd=None):
    global cnxpool
    print("{} initializing mysql connection pool...".format(
        strftime("%H:%M:%S")))
    cnxpool = MySQLConnectionPool(
        pool_name="dbpool",
        pool_size=db_pool_size or 5,
        host=db_host or '127.0.0.1',
        port=db_port or 3306,
        user='mysql',
        database='secu',
        password=db_pwd or '123456',
        # ssl_ca='',
        # use_pure=True,
        connect_timeout=90000)

def _impute():
    global cnxpool
    c = cnxpool.get_connection()
    query = 'select code, date, amount, xrate, close, high, high_close, open, open_close, low, low_close, volume from index_d_n_lr'
    df = pd.read_sql(query, c)

if __name__ == '__main__':
    args = _parseArgs()
    _init(4, 
        db_host=args.db_host,
        db_port=args.db_port, 
        db_pwd=args.db_pwd
    )
    _impute()