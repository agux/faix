import tensorflow as tf
import numpy as np
import os
import sys

np.set_printoptions(threshold=np.inf,
                    suppress=True,
                    formatter={'float': '{: 0.5f}'.format})


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

inputs = np.ones((5,5))
total_steps = 50
print(inputs)
dropout = DecayedDropoutLayer(dropout="dropout",
                    decay_start=10,
                    initial_dropout_rate=0.5,
                    first_decay_steps=10,
                    t_mul=1.05,
                    m_mul=0.98,
                    alpha=0.01,
                    seed=1234
                )

for _ in range(total_steps):
    output=dropout(inputs)
    print(output.numpy())