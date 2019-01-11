import math

import tensorflow as tf

import numpy as np

from .configs import *
from base.dataset import DatasetManager
from base.rprop import RPropOptimizer


def init_models_for_pre_train(batch_manager):
    def _create_p_or_q_variable(n, rank, batch_manager):
        # TODO: 밖으로 꺼내야 함.
        mu = batch_manager.mu
        std = batch_manager.std

        _mu = math.sqrt(mu / rank)
        _std = math.sqrt((math.sqrt(mu * mu + std * std) - mu) / rank)
        return tf.Variable(tf.truncated_normal([n, rank], _mu, _std))

    n_row, n_col = batch_manager.n_user, batch_manager.n_item

    u = tf.placeholder(tf.int64, [None], name='u')
    i = tf.placeholder(tf.int64, [None], name='i')
    r = tf.placeholder(tf.float32, [None], name='r')

    # init weights
    mu = batch_manager.mu
    std = batch_manager.std
    p = _create_p_or_q_variable(n_row, PRE_RANK, batch_manager)
    q = _create_p_or_q_variable(n_col, PRE_RANK, batch_manager)

    p_lookup = tf.nn.embedding_lookup(p, u)
    q_lookup = tf.nn.embedding_lookup(q, i)
    r_hat = tf.reduce_sum(tf.multiply(p_lookup, q_lookup), 1)

    reg_loss = tf.add_n(
        [tf.reduce_sum(tf.square(p)),
         tf.reduce_sum(tf.square(q))])
    loss = tf.reduce_sum(tf.square(r - r_hat)) + PRE_LAMBDA * reg_loss
    rmse = tf.sqrt(tf.reduce_mean(tf.square(r - r_hat)))

    optimizer = tf.train.MomentumOptimizer(PRE_LEARNING_RATE, 0.9)
    # optimizer = tf.train.GradientDescentOptimizer(PRE_LEARNING_RATE)
    train_ops = [
        optimizer.minimize(loss, var_list=[p]),
        optimizer.minimize(loss, var_list=[q])
    ]

    return {
        'u': u,
        'i': i,
        'r': r,
        'train_ops': train_ops,
        'loss': loss,
        'rmse': rmse,
        'p': p,
        'q': q,
    }


def init_models(batch_manager):
    n_row, n_col = batch_manager.n_user, batch_manager.n_item

    u = tf.placeholder(tf.int64, [None], name='u')
    i = tf.placeholder(tf.int64, [None], name='i')
    r = tf.placeholder(tf.float32, [None], name='r')
    k = tf.placeholder(tf.float32, [None], name='k')

    # train_u = tf.constant(
    #     batch_manager.train_data[:, 0], dtype=tf.int64, name='train_u')
    # train_i = tf.constant(
    #     batch_manager.train_data[:, 1], dtype=tf.int64, name='train_i')
    # train_r = tf.constant(
    #     batch_manager.train_data[:, 2], dtype=tf.float32, name='train_r')

    # row_vecs = tf.SparseTensor(
    #     indices=tf.stack([train_u, train_i], axis=1),
    #     values=train_r,
    #     dense_shape=[n_row, n_col])
    # col_vecs = tf.SparseTensor(
    #     indices=tf.stack([train_i, train_u], axis=1),
    #     values=train_r,
    #     dense_shape=[n_col, n_row])

    # init weights
    local_p = tf.Variable(tf.zeros([n_row, LOCAL_RANK]))
    local_q = tf.Variable(tf.zeros([n_col, LOCAL_RANK]))

    local_p_lookup = tf.nn.embedding_lookup(local_p, u)
    local_q_lookup = tf.nn.embedding_lookup(local_q, i)
    local_r_hat = tf.reduce_sum(
        tf.multiply(local_p_lookup, local_q_lookup), axis=1)
    local_loss = tf.reduce_sum(
        tf.square(r - local_r_hat) * k) + LOCAL_LAMBDA * tf.add_n([
            tf.reduce_sum(tf.square(local_p)),
            tf.reduce_sum(tf.square(local_q))
        ])
    local_sse = tf.reduce_sum(tf.square(r - local_r_hat))

    # local_optimizer = tf.train.MomentumOptimizer(LOCAL_LEARNING_RATE, 0.9)
    # _optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9)
    # _optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    local_optimizer = tf.train.GradientDescentOptimizer(LOCAL_LEARNING_RATE)
    local_train_op = local_optimizer.minimize(
        local_loss, var_list=[local_p, local_q])

    return {
        'u': u,
        'i': i,
        'r': r,
        'k': k,
        'local_p': local_p,
        'local_q': local_q,
        'local_r_hat': local_r_hat,
        'local_train_op': local_train_op,
        'local_sse': local_sse,
    }
