
import tensorflow as tf

from tensorflow import keras
from time import strftime

class DelayedCosineDecayRestarts(keras.experimental.CosineDecayRestarts):

    def __init__(self, decay_start, *args, **kwargs):
        super(DelayedCosineDecayRestarts, self).__init__(*args, **kwargs)
        self._decay_start = decay_start
        self.started = False

    def __call__(self, step):
        return tf.cond(
            tf.less(step, self._decay_start), 
            lambda: self.initial_learning_rate,
            lambda: self.decay(step)
        )

    def decay(self, step):
        if step >= self._decay_start and not self.started:
            print("DelayedCosineDecayRestarts activated at step {}".format(step))
            self.started = True
        return super(DelayedCosineDecayRestarts, self).__call__(step-self._decay_start+1)

    def get_config(self):
        return {
            "decay_start": self._decay_start
        }