import numpy as np
import tensorflow as tf

feature_columns = [tf.feature_column.numeric_column('x', shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns)

train_x = np.array([1., 2., 3., 4.])
train_y = np.array([0., -1., -2., -3.])
eval_x = np.array([2.,5.,8.,1.])
eval_y = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn({'x':train_x}, train_y, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({'x':train_x}, train_y, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x':eval_x}, eval_y, batch_size=4, num_epochs=1000, shuffle=False)

estimator.train(input_fn, steps=1000)

train_metrics = estimator.evaluate(train_input_fn)
eval_metrics = estimator.evaluate(eval_input_fn)

print('train metrics: %r'%train_metrics)
print('eval metrics: %r'%eval_metrics)