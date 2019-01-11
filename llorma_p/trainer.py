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


def __get_rmse(local_models, batch_manager, key='train'):
    r_hats = np.stack(
        [
            getattr(local_model, '{}_r_hat'.format(key))
            for local_model in local_models
        ],
        axis=1)
    ks = np.stack(
        [
            getattr(local_model, '{}_k'.format(key))
            for local_model in local_models
        ],
        axis=1)
    sum_ks = np.sum(ks, axis=1)
    sum_r_hats = np.sum(np.multiply(r_hats, ks), axis=1)
    r_hat = tf.divide(sum_r_hats, sum_ks)
    r_hat[np.isnan(r_hat)] = 3

    r = getattr(batch_manager, '{}_data'.format(key))[:, 2]
    rmse = np.sqrt(np.mean(np.square(r_hat - r)))
    return rmse


def _get_rmses(local_models, batch_manager):
    train_rmse = __get_rmse(local_models, batch_manager, key='train')
    test_rmse = __get_rmse(local_models, batch_manager, key='test')
    return train_rmse, test_rmse


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

    for local_idx, local_model in enumerate(local_models):
        local_model.train()

        train_rmse, test_rmse = _get_rmses(local_models[:i+1], batch_manager)

        print(">> LOCAL [{:3d}] {:.5f}, {:.5f}\n".format(local_idx, train_rmse,
                                                  test_rmse))


def main(kind):
    _train(kind)
