from __future__ import print_function
import pandas as pd
import mysql.connector
import numpy as np
from pstk import data as dat
from sqlalchemy import create_engine


data = dat.loadPrepTrainingData4D(1, 3)
print(np.array(data[1]).shape)


engine = create_engine('mysql+mysqlconnector://mysql:123456@localhost/secu')

df = pd.read_sql_query(
    "SELECT code, date, open,high,low,close,volume FROM kline_d_b where date between '1995-01-05' and '1995-01-06'",
    engine,
    index_col=["code", "date"])
try:
    print(df.loc["000004"].loc["1995-01-06"].as_matrix())
    print(df.index.get_level_values('code').tolist()[0])
except KeyError:
    print("key not found")
except Exception:
    raise
