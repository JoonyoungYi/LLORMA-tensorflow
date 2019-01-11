import math

import tensorflow as tf


def cosine_decay_learning_rate(learning_rate,
                               global_step,
                               decay_steps=200,
                               alpha=0.01):
    # tensorflow==1.4.0에서 못쓰니까 구현.
    global_step = tf.cast(global_step, tf.int64)
    step = tf.cast(tf.mod(global_step, decay_steps), tf.float32)
    cosine_decay = 0.5 * (1.0 + tf.cos(math.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return learning_rate * decayed
