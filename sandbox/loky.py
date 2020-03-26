import os
import loky
from time import sleep
# from loky import get_reusable_executor


def say_hello(k):
    pid = os.getpid()
    print("Hello from {} with arg {}".format(pid, k))
    sleep(.01)
    return pid


# Create an executor with 4 worker processes, that will
# automatically shutdown after idling for 2s
executor = loky.get_reusable_executor(max_workers=4, timeout=2)

res = executor.submit(say_hello, 1)
print("Got results:", res.result())

results = executor.map(say_hello, range(50))
n_workers = len(set(results))
print("Number of used processes:", n_workers)
assert n_workers == 4