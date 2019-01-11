import math

import tensorflow as tf

import numpy as np

from .configs import *
from base.dataset import DatasetManager
from base.rprop import RPropOptimizer


def _create_p_or_q_variable(n, rank, batch_manager):
    mu = batch_manager.mu
    std = batch_manager.std

    _mu = math.sqrt(mu / rank)
    _std = math.sqrt((math.sqrt(mu * mu + std * std) - mu) / rank)
    return tf.Variable(
        tf.truncated_normal([n, rank], _mu, _std, dtype=tf.float64))


def init_models_for_pre_train(batch_manager):
    n_row, n_col = batch_manager.n_user, batch_manager.n_item

    u = tf.placeholder(tf.int64, [None], name='u')
    i = tf.placeholder(tf.int64, [None], name='i')
    r = tf.placeholder(tf.float64, [None], name='r')

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


def _get_train_op(optimizer, loss, var_list):
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    # capped_gvs = [(tf.clip_by_value(grad, -100.0, 100.0), var)
    #               for grad, var in gvs]
    capped_gvs = gvs
    train_op = optimizer.apply_gradients(capped_gvs)
    return train_op


def init_models(batch_manager):
    n_row, n_col = batch_manager.n_user, batch_manager.n_item

    u = tf.placeholder(tf.int64, [None], name='u')
    i = tf.placeholder(tf.int64, [None], name='i')
    r = tf.placeholder(tf.float64, [None], name='r')
    k = tf.placeholder(tf.float64, [None, N_ANCHOR], name='k')
    k_sum = tf.reduce_sum(k, axis=1)

    # init weights
    ps, qs, losses, r_hats = [], [], [], []
    for anchor_idx in range(N_ANCHOR):
        p = _create_p_or_q_variable(n_row, RANK, batch_manager)
        q = _create_p_or_q_variable(n_col, RANK, batch_manager)
        ps.append(p)
        qs.append(q)

        p_lookup = tf.nn.embedding_lookup(p, u)
        q_lookup = tf.nn.embedding_lookup(q, i)
        r_hat = tf.reduce_sum(tf.multiply(p_lookup, q_lookup), axis=1)
        r_hats.append(r_hat)

    r_hat = tf.reduce_sum(tf.multiply(k, tf.stack(r_hats, axis=1)), axis=1)
    r_hat = tf.where(tf.greater(k_sum, 1e-2), r_hat, tf.ones_like(r_hat) * 3)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(r - r_hat)))

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    loss = tf.reduce_sum(tf.square(r_hat - r)) + LAMBDA * tf.reduce_sum(
        [tf.reduce_sum(tf.square(p_or_q)) for p_or_q in ps + qs])
    train_ops = [
        _get_train_op(optimizer, loss, [p, q]) for p, q in zip(ps, qs)
    ]

    return {
        'u': u,
        'i': i,
        'r': r,
        'k': k,
        'train_ops': train_ops,
        'rmse': rmse,
    }
