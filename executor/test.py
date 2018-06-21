from loky import get_reusable_executor

_executor = get_reusable_executor(
    max_workers=2, timeout=20)


def fn(p):
    a, b, c = p
    print("received:{} {} {}".format(a, b, c))
    return p[2]+1, [[p[0]+1, p[0]+2], [p[0]+3, p[0]+4]], p[1]+1


params = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
r = list(_executor.map(fn, params))
ra, rb, rc = zip(*r)
print("a:{}".format(ra))
print("b:{}".format(rb))
print("c:{}".format(rc))
