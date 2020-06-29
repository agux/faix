import tensorflow as tf
import numpy as np
import os
import sys


def setupPath():
    p1 = os.path.dirname(os.path.abspath(__file__))
    p2 = os.path.dirname(p1)
    p3 = os.path.dirname(p2)
    p4 = os.path.dirname(p3)
    sys.path.append([p1,p2,p3,p4])
    # os.environ[
    #     "PYTHONPATH"] = p1 + ":" + p2 + ":" + p3 + ":" + p4 + ":" + os.environ.get(
    #         "PYTHONPATH", "")
    print(sys.path)

# setupPath()

from corl.model.tf2.common import DecayedDropoutLayer

# print(dir(tf2))

inputs = np.random.sample((10,10))
total_steps = 50
print(inputs)
dropout = DecayedDropoutLayer(dropout="dropout",
                    decay_start=10,
                    initial_dropout_rate=0.5,
                    first_decay_steps=10,
                    t_mul=2.0,
                    m_mul=1.0,
                    alpha=0.0,
                    seed=1234
                )

for _ in range(total_steps):
    dropout(inputs)