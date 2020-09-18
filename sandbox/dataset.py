import tensorflow as tf

d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])


# transform a string tensor to upper case string using a Python function
def upper_case_fn(t: tf.Tensor):
    return t.numpy().decode('utf-8').upper()


d = d.map(
    lambda x: tf.py_function(func=upper_case_fn, inp=[x], Tout=tf.string))

print(list(d.as_numpy_iterator()))

a = [1,2,3,4]

