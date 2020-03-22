import numpy as np
import tensorflow as tf

def model_fn(features, labels, mode):
    W = tf.compat.v1.get_variable("W",[1],tf.float64)
    b = tf.compat.v1.get_variable("b",[1],tf.float64)
    y = W*features["x"] + b
    loss = tf.reduce_sum(input_tensor=tf.square(y-labels))
    global_step = tf.compat.v1.train.get_global_step()
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(.01)
    train = tf.group(
        optimizer.minimize(loss),
        tf.compat.v1.assign_add(global_step, 1)
    )
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train)


x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7., 0.])

input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

estimator = tf.estimator.Estimator(model_fn=model_fn)
estimator.train(input_fn=input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r"%train_metrics)
print("eval metrics: %r"%eval_metrics)