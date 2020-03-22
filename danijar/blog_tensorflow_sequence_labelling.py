# Example for danijar blog post at:
# http://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
import functools
import sets
import tensorflow as tf


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceLabelling:

    def __init__(self, data, target, dropout, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        network = tf.compat.v1.nn.rnn_cell.GRUCell(self._num_hidden)
        network = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            network, output_keep_prob=self.dropout)
        network = tf.compat.v1.nn.rnn_cell.MultiRNNCell([network] * self._num_layers)
        output, _ = tf.compat.v1.nn.dynamic_rnn(network, data, dtype=tf.float32)
        # Softmax layer.
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self._num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(
            input_tensor=self.target * tf.math.log(self.prediction), axis=[1, 2])
        cross_entropy = tf.reduce_mean(input_tensor=cross_entropy)
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(input=self.target, axis=2), tf.argmax(input=self.prediction, axis=2))
        return tf.reduce_mean(input_tensor=tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.random.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def read_dataset():
    dataset = sets.Ocr()
    dataset = sets.OneHot(dataset.target, depth=2)(dataset, columns=['target'])
    dataset['data'] = dataset.data.reshape(
        dataset.data.shape[:-2] + (-1,)).astype(float)
    train, test = sets.Split(0.66)(dataset)
    return train, test


if __name__ == '__main__':
    train, test = read_dataset()
    _, length, image_size = train.data.shape
    num_classes = train.target.shape[2]
    data = tf.compat.v1.placeholder(tf.float32, [None, length, image_size])
    target = tf.compat.v1.placeholder(tf.float32, [None, length, num_classes])
    dropout = tf.compat.v1.placeholder(tf.float32)
    model = SequenceLabelling(data, target, dropout)
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.initialize_all_variables())
    for epoch in range(10):
        for _ in range(100):
            batch = train.sample(10)
            sess.run(model.optimize, {
                data: batch.data, target: batch.target, dropout: 0.5})
        error = sess.run(model.error, {
            data: test.data, target: test.target, dropout: 1})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
