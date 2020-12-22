

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
