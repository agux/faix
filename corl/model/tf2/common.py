
import tensorflow as tf

from tensorflow import keras
from time import strftime

class DelayedCosineDecayRestarts(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, decay_start, *args, **kwargs):
        super(DelayedCosineDecayRestarts, self).__init__(**kwargs)
        self._decay_start = decay_start
        self.cdr = keras.experimental.CosineDecayRestarts(*args, **kwargs)

    def __call__(self, step):
        return tf.cond(
            tf.less(step, self._decay_start),
            self.initial_learning_rate,
            self.cdr(step-self._decay_start+1)
        )
    
    def get_config(self):
        return {
            "decay_start": self._decay_start
        }