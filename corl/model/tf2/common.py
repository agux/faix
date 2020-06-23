
import tensorflow as tf

from tensorflow import keras
from time import strftime

class DelayedCosineDecayRestarts(keras.experimental.CosineDecayRestarts):

    def __init__(self, decay_start, *args, **kwargs):
        super(DelayedCosineDecayRestarts, self).__init__(*args, **kwargs)
        self._decay_start = decay_start

    def __call__(self, step):
        return tf.cond(
            tf.less(step, self._decay_start), 
            lambda: self.initial_learning_rate,
            lambda: self.decay(step)
        )

    def decay(self, step):
        lr = super(DelayedCosineDecayRestarts, self).__call__(step-self._decay_start+1)
        tf.print("DelayedCosineDecayRestarts activated at step: ", step, ", lr= ", lr)
        return lr

    def get_config(self):
        return {
            "decay_start": self._decay_start
        }