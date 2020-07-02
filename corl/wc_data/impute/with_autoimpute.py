import pandas as pd
import numpy as np
from time import strftime
from mysql.connector.pooling import MySQLConnectionPool
from corl.wc_data.impute.common import parseArgs, init, print_header
from autoimpute.imputations import SingleImputer, MultipleImputer, MiceImputer

import warnings
warnings.filterwarnings("ignore")

cnxpool = None


def _impute():
    c = cnxpool.get_connection()
    query = 'select code, date, amount, xrate, close, high, high_close, open, open_close, low, low_close, volume from index_d_n_lr order by code asc, date asc'
    df = pd.read_sql(query, c)
    print_header('original table:')
    print(df)
    print_header('Rows having NaN:')
    nan_df = df[df.isna().any(axis=1)]
    print(nan_df)
    sdf = df[['amount', 'xrate', 'close', 'high', 'high_close',
              'open', 'open_close', 'low', 'low_close', 'volume']]
    print_header('Subset of Table')
    print(sdf)
    mi = MiceImputer()
    mi_data_full = mi.fit_transform(sdf)

    for i, m in enumerate(mi_data_full):
        print_header("After Imputation #{}".format(i+1))
        imputed_filtered = m[[m.isna().any(axis=1)]]
        print(imputed_filtered)

    # print the results
    # print_header("Results from SingleImputer running PMM on column y one time")
    # conc = pd.concat([data_miss.head(20), si_data_full.head(20)], axis=1)
    # conc.columns = ["x", "y_orig", "x_imp", "y_imp"]
    # conc[["x", "y_orig", "y_imp"]]


if __name__ == '__main__':
    args = parseArgs()
    cnxpool = init(4,
                   db_host=args.db_host,
                   db_port=args.db_port,
                   db_pwd=args.db_pwd
                   )
    _impute()
