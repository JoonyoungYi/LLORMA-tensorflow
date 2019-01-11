"""
    RProp (Resilient Backpropagation) for TensorFlow.
    This code is forked form "https://raw.githubusercontent.com/dirkweissenborn/genie-kb/master/rprop.py".
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class RPropOptimizer(optimizer.Optimizer):
    """
        Optimizer that implements the RProp algorithm.
    """

    def __init__(self,
                 stepsize=0.1,
                 etaplus=1.2,
                 etaminus=0.5,
                 stepsizemax=50.0,
                 stepsizemin=1e-6,
                 use_locking=False,
                 name="RProp"):
        super(RPropOptimizer, self).__init__(use_locking, name)
        self._stepsize = stepsize
        self._etaplus = etaplus
        self._etaminus = etaminus
        self._stepsizemax = stepsizemax
        self._stepsizemin = stepsizemin

    def _create_slots(self, var_list):
        '''
        :param var_list:
        :return:
        '''
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable.

        # Create slots for the first and second moments.
        for v in var_list:
            self._get_or_make_slot(
                v,
                tf.ones([v.get_shape().num_elements()], dtype=tf.float32) *
                self._stepsize,
                "step",
                self._name, )
            self._get_or_make_slot(
                v,
                tf.zeros([v.get_shape().num_elements()], dtype=tf.float32),
                "delta",
                self._name, )
            self._get_or_make_slot(
                v,
                tf.zeros([v.get_shape().num_elements()], dtype=tf.float32),
                "grad",
                self._name, )

    def _apply_dense(self, grad, var):
        grad_slot = self.get_slot(var, "grad")
        step_slot = self.get_slot(var, "step")
        delta_slot = self.get_slot(var, "delta")

        grad = tf.reshape(grad, [-1])
        sign = tf.cast(tf.sign(grad_slot * grad), tf.int64)
        with tf.control_dependencies([sign]):
            grad = grad_slot.assign(grad)

            p_indices = tf.where(tf.equal(sign, 1))  # positive indices
            m_indices = tf.where(tf.equal(sign, -1))  # minus indices
            z_indices = tf.where(tf.equal(sign, 0))  # zero indices

        step_p_update = tf.expand_dims(
            tf.minimum(
                tf.gather_nd(step_slot, p_indices) * self._etaplus,
                self._stepsizemax), 1)
        step_m_update = tf.expand_dims(
            tf.maximum(
                tf.gather_nd(step_slot, m_indices) * self._etaminus,
                self._stepsizemin), 1)
        step_z_update = tf.expand_dims(tf.gather_nd(step_slot, z_indices), 1)
        with tf.control_dependencies(
            [step_p_update, step_m_update, step_z_update]):
            step = tf.scatter_update(step_slot, p_indices, step_p_update)
            step = tf.scatter_update(step, m_indices, step_m_update)
            step = tf.scatter_update(step, z_indices, step_z_update)
            step = step_slot.assign(step)

        delta_p_update = tf.expand_dims(
            tf.gather_nd(tf.sign(grad) * step, p_indices), 1)
        delta_z_update = tf.expand_dims(
            tf.gather_nd(tf.sign(grad) * step, z_indices), 1)
        with tf.control_dependencies([delta_p_update, delta_z_update]):
            delta = tf.scatter_update(delta_slot, p_indices, delta_p_update)
            delta = tf.scatter_update(delta, z_indices, delta_z_update)
            delta = delta_slot.assign(delta)

        with tf.control_dependencies([sign]):
            grad = tf.scatter_update(grad, m_indices,
                                     tf.zeros_like(m_indices, tf.float32))
            grad = grad_slot.assign(grad)

        up = tf.reshape(delta, var.get_shape())
        var_update = var.assign_sub(up, use_locking=self._use_locking)

        return tf.group(*[var_update, step, delta, grad])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("RProp should be used only in batch_mode.")
