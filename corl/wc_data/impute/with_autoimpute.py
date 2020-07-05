import pandas as pd
import numpy as np
from time import strftime
from mysql.connector.pooling import MySQLConnectionPool
from corl.wc_data.impute.common import parseArgs, init, print_header
from autoimpute.imputations import SingleImputer, MultipleImputer, MiceImputer

import warnings
warnings.filterwarnings("ignore")

cnxpool = None


def _tutorial():
    # imports
    from scipy.stats import norm, binom
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(context="talk", rc={'figure.figsize': (11.7, 8.27)})

    # seed to follow along
    np.random.seed(654654)

    # generate 1500 data points
    N = np.arange(1500)

    # helper function for this data
    def vary(v): return np.random.choice(np.arange(v))

    # create correlated, random variables
    a = 2
    b = 1/2
    eps = np.array([norm(0, vary(50)).rvs() for n in N])
    y = (a + b*N + eps) / 100
    x = (N + norm(10, vary(250)).rvs(len(N))) / 100
    z = y * a + vary(1000) * b**2 - x*y

    # 20% missing in x, 30% missing in y
    x[binom(1, 0.2).rvs(len(N)) == 1] = np.nan
    y[binom(1, 0.3).rvs(len(N)) == 1] = np.nan
    z[binom(1, 0.4).rvs(len(N)) == 1] = np.nan

    # collect results in a dataframe
    data_miss = pd.DataFrame({"m": y, "p": x, "q": z})
    # sns.scatterplot(x="x", y="y", data=data_miss)
    # plt.show()

    # The plot suggests a linear relationship may exist between x and y. Let's fit a linear model to estimate that relationship.
    # from sklearn.linear_model import LinearRegression

    # # prep for regression
    # X = data_miss.x.values.reshape(-1, 1) # reshape because one feature only
    # y = data_miss.y
    # lm = LinearRegression()

    # # try to fit the model
    # print_header("Fitting linear model to estimate relationship between X and y")
    # try:
    #     lm.fit(X, y)
    # except ValueError as ve:
    #     print(f"{ve.__class__.__name__}: {ve}")

    # amount of missing data before imputation
    print_header("Amount of data missing before imputation takes place")
    print(pd.DataFrame(data_miss.isnull().sum(),
                       columns=["records missing"]).T)

    print_header(
        "Imputing missing data in one line of code with the default SingleImputer")
    data_imputed_once = SingleImputer().fit_transform(data_miss)
    print("Imputation Successful!")

    # amount of missing data before imputation
    print_header("Amount of data missing after imputation takes place")
    print(pd.DataFrame(data_imputed_once.isnull().sum(),
                       columns=["records missing"]).T)


def _impute():
    c = cnxpool.get_connection()
    query = 'select code, date, amount, xrate, close, high, high_close, open, open_close, low, low_close, volume from index_d_n_lr order by code asc, date asc'
    df = pd.read_sql(query, c)
    print('original table has {} rows'.format(len(df.index)))
    # print(df)

    print_header('Code Set')
    code_set = df.drop_duplicates(["code"])
    print(code_set)

    print_header('Volume rows having NaN:')
    nan_df = df[df['volume'].isna()]
    print(nan_df)

    # collect code -> [missing dates] mapping
    nan_code_set = nan_df.drop_duplicates(["code"])
    missing_dates = {}
    for code in nan_code_set["code"].values.tolist():
        missing_dates[code] = nan_df[nan_df['code']==code]['date'].values.tolist()

    # print_header('Codes having NaN:')
    # unique_codes = nan_df.drop_duplicates(["code"])
    # print(unique_codes)

    # print_header('DJI:')
    # dji_df = df[df['code']=='.DJI']
    # print(dji_df)

    # print_header('Subset of Table')
    # sdf = df[[
    #     'code',
    #     'date',
    #     # 'amount',
    #     # 'xrate',
    #     'close',
    #     'high',
    #     'high_close',
    #     'open',
    #     'open_close',
    #     'low',
    #     'low_close',
    #     'volume']]
    # print(sdf)

    # sdf = dji_df[['amount', 'xrate', 'close', 'high', 'high_close',
    #           'open', 'open_close', 'low', 'low_close', 'volume']]
    # print(sdf)

    # For Single Imputer
    impt = MiceImputer(
        strategy={
            "volume": 'default predictive',
            # "volume": "default time",
        },
        predictors={
            "volume": [
                "close",
                "high",
                "high_close",
                "open",
                "open_close",
                "low",
                "low_close"
            ],
        },
        return_list=True
    )

    # df_imputed = impt.fit_transform(sdf)

    # fit the model on code basis
    for code in code_set["code"].values.tolist():
        code_df = df[df['code'] == code]
        print("fitting on code: {}, num records: {}".format(
            code, len(code_df.index)))
        impt.fit(code_df)
        
    # transform (impute) the missing data
    for code in nan_code_set["code"].values.tolist():
        code_df = df[df['code'] == code]
        print_header("transforming for code: {}, num records: {}".format(
            code, len(code_df.index)))
        imputed_df = impt.transform(code_df)
        print_header("After Imputation for {}".format(code))
        for i, m in imputed_df:
            print_header("result#{}".format(i))
            imputed_filtered = m[m['date'].isin(missing_dates[code])]
            print(imputed_filtered)

    # print_header("After Imputation")
    # print(si_data_full)
    # imputed_filtered = df_imputed[(df_imputed['code'] == 'HKVHSI')
    # #                                 & (df_imputed['date'].isin(
    # #                                     ['2018-01-11', '2018-01-12']))
    # # ]
    # print(imputed_filtered)

    # print_header("After Imputation")
    # for i, m in df_imputed:
    #     print_header("Result #{}".format(i))
    #     # print(m)
    #     # m = pd.DataFrame.from_dict(dict(m))
    #     imputed_filtered = m[(m['code'] == 'HKVHSI') & (m['date'].isin(
    #         ['2018-01-11', '2018-01-12']))]
    #     print(imputed_filtered)

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

    # _tutorial()
