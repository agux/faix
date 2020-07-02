import argparse
from time import strftime
from mysql.connector.pooling import MySQLConnectionPool

print_header = lambda msg: print(f"{msg}\n{'-'*len(msg)}")

cnxpool = None

def parseArgs():
    parser = argparse.ArgumentParser(add_help=False)
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

def init(db_pool_size=None, db_host=None, db_port=None, db_pwd=None):
    global cnxpool
    if cnxpool is not None:
        return cnxpool
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
    return cnxpool