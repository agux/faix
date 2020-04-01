import ray
from mysql.connector.pooling import MySQLConnectionPool

# ray.init(num_cpus=16, memory=17179869184, object_store_memory=8589934592)
ray.init()


# @ray.remote(num_cpus=0.01)
@ray.remote
def f(x, s):
    cnxpool = MySQLConnectionPool(
        pool_name="dbpool",
        pool_size=5,
        host='10.32.1.31',
        port=3306,
        user='mysql',
        database='secu',
        password='rap!It6Q=p!aS6z3PRl0',
        # ssl_ca='',
        # use_pure=True,
        connect_timeout=90000)
    return x * x, x + 1, s['a'] * s['b']


shared = ray.put({'a': 99, 'b': 100})

futures = [f.remote(i, shared) for i in range(4)]

r = list(ray.get(futures))

a, b, c = zip(*r)

print(a)
print(b)
print(c)