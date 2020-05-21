import tensorflow as tf
import numpy as np


def step(pre, inputs):
    print("pre:{}, inputs:{}".format(pre, inputs))
    el1, el2 = inputs
    pel1, pel2 = pre
    return (el1 + pel1, el2 + pel2)


elems1 = np.array([1, 2, 3, 4, 5, 6])
elems2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
inputs = (elems1, elems2)
sum = tf.scan(step, inputs)
print(sum)
# sum == [1, 3, 6, 10, 15, 21]