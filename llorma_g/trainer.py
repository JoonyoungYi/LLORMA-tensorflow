import os
import time
import math
import random

import tensorflow as tf
import numpy as np

from . import pre_trainer
from .anchor import AnchorManager
from .batch import BatchManager
from .configs import *
from .local import LocalModel
from .model import init_models


def __init_session():
    # gpu_options = tf.GPUOptions(
    #     per_process_gpu_memory_fraction=GPU_MEMORY_FRAC)
    # gpu_config = tf.ConfigProto(gpu_options=gpu_options)
    # session = tf.Session(config=gpu_config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    return session


def _get_k(local_models, kind='train'):
    k = np.stack(
        [
            getattr(local_model, '{}_k'.format(kind))
            for local_model in local_models
        ],
        axis=1)
    k = np.clip(k, 0.0, 1.0)
    k = np.divide(k, np.sum(k, axis=1, keepdims=1))
    k[np.isnan(k)] = 0
    return k


def _validate(
        session,
        models,
        batch_manager,
        valid_k,
        test_k, ):
    valid_rmse = session.run(
        models['rmse'],
        feed_dict={
            models['u']: batch_manager.valid_data[:, 0],
            models['i']: batch_manager.valid_data[:, 1],
            models['r']: batch_manager.valid_data[:, 2],
            models['k']: valid_k,
        })

    test_rmse = session.run(
        models['rmse'],
        feed_dict={
            models['u']: batch_manager.test_data[:, 0],
            models['i']: batch_manager.test_data[:, 1],
            models['r']: batch_manager.test_data[:, 2],
            models['k']: test_k,
        })

    return valid_rmse, test_rmse


def _train(kind):
    row_latent_init, col_latent_init = pre_trainer.get_p_and_q(
        kind, use_cache=USE_CACHE)

    batch_manager = BatchManager(kind)
    models = init_models(batch_manager)

    session = __init_session()
    anchor_manager = AnchorManager(
        session,
        models,
        batch_manager,
        row_latent_init,
        col_latent_init, )
    local_models = [
        LocalModel(session, models, anchor_idx, anchor_manager, batch_manager)
        for anchor_idx in range(N_ANCHOR)
    ]

    train_k = _get_k(local_models, kind='train')
    valid_k = _get_k(local_models, kind='valid')
    test_k = _get_k(local_models, kind='test')

    min_valid_rmse = float("Inf")
    min_valid_iter = 0
    final_test_rmse = float("Inf")
    start_time = time.time()

    batch_rmses = []
    train_data = batch_manager.train_data

    for iter in range(10000000):
        for m in range(0, train_data.shape[0], BATCH_SIZE):
            end_m = min(m + BATCH_SIZE, train_data.shape[0])
            u = train_data[m:end_m, 0]
            i = train_data[m:end_m, 1]
            r = train_data[m:end_m, 2]
            k = train_k[m:end_m, :]
            results = session.run(
                [models['rmse']] + models['train_ops'],
                feed_dict={
                    models['u']: u,
                    models['i']: i,
                    models['r']: r,
                    models['k']: k,
                })
            batch_rmses.append(results[0])

            if m % (BATCH_SIZE * 100) == 0:
                print('  - ', results[:1])

        if iter % 1 == 0:
            valid_rmse, test_rmse = _validate(session, models, batch_manager,
                                              valid_k, test_k)
            if valid_rmse < min_valid_rmse:
                min_valid_rmse = valid_rmse
                min_valid_iter = iter
                final_test_rmse = test_rmse

            batch_rmse = sum(batch_rmses) / len(batch_rmses)
            batch_rmses = []
            print('  - ITER{:4d}:'.format(iter),
                  "{:.5f}, {:.5f} {:.5f} / {:.5f}".format(
                      batch_rmse, valid_rmse, test_rmse, final_test_rmse))


def main(kind):
    _train(kind)
