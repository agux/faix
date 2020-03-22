import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
line_model = W*x+b

loss = tf.reduce_sum(input_tensor=tf.square(line_model-y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(.01)
train = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})

curr_W, curr_b, curr_loss = sess.run([W,b,loss],{x:x_train, y:y_train})
print("W: %s   b: %s   loss: %s"%(curr_W,curr_b,curr_loss))

fw = tf.compat.v1.summary.FileWriter('logdir', graph=sess.graph)
fw.flush()
fw.close()