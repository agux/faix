import tensorflow as tf

from tensorflow.python import pywrap_tfe as pywrap_tfe

# i = tf.keras.Input(shape=[2, 3])
# s = tf.shape(i)
# f = tf.fill([s[0], 4,5], EPSILON, name="f")
# f2 = tf.fill([s[0], 2], EPSILON, name="f2")
# c = tf.constant(EPSILON, shape=[s[0], 2, 2])
# c = tf.constant_initializer(EPSILON)(shape=[None, 2,2])
# z = tf.zeros([s[0], 2], name="aaa")
# z2 = tf.
# print("None: {}".format(s[0]))
# print("fill: {}".format(f))
# print("constant: {}".format(c))
# print(f)
# print(f2)
cell = tf.keras.layers.LSTMCell(units=32, name="controller")
# state = cell.get_initial_state(batch_size=s[0], dtype=tf.float32)
i = tf.keras.Input(shape=[2, 3])
state = cell.get_initial_state(batch_size=tf.shape(i)[0], dtype=tf.float32)
print(state)

pywrap_tfe.TFE_Py_FastPathExecute()
