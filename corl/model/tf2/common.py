
import tensorflow as tf

from tensorflow import keras
from time import strftime

class DelayedCosineDecayRestarts(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, decay_start, first_decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0,
        name=None, *args, **kwargs):

        super(DelayedCosineDecayRestarts, self).__init__()
        self.initial_learning_rate=initial_learning_rate
        self.decay_start = decay_start
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        self.name = name
        self.cdr = keras.experimental.CosineDecayRestarts(
            initial_learning_rate, 
            first_decay_steps, 
            t_mul, 
            m_mul, 
            alpha,
            name+"_cdr")

    def __call__(self, step):
        return tf.cond(
            tf.less(step, self.decay_start),
            self.initial_learning_rate,
            self.cdr(step-self.decay_start+1)
        )
    
    def get_config(self):
        return {
            "initial_learning_rate": initial_learning_rate,
            "decay_start": self.decay_start,
            "first_decay_steps": first_decay_steps,
            "t_mul": t_mul,
            "m_mul": m_mul,
            "alpha": alpha,
            "name": name
        }