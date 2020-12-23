

d = '2001-12-31'
s, e, _ = d.split('-')
s, e = int(s), int(e)
print(s)
print(e)


def get_ymstr(date_start, date_end):
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


print(get_ymstr('2020-02-01', '2020-12-20'))

import numpy as np
import pandas as pd

offset = 38

code = ['000001', '300013', '613002', '002131', '000001']
date = ['2020-01-01', '2019-12-10', '2018-05-09', '2017-11-30', '2020-12-31']
klid = [100, 300, 400, 500, 99]

df = pd.DataFrame([code, date, klid], index=['code', 'date', 'klid']).T

tp = list(zip(code, date, np.array(klid)-offset))

print(tp)
print(df)

sorted_tp = sorted(tp, key=lambda tup: (tup[0], tup[2]))

print(sorted_tp)



