import os
import time
import math
import random

import tensorflow as tf
import numpy as np

from .batch import BatchManager
from .configs import *
from .model import init_models_for_pre_train


def _validate(session, batch_manager, models):
    valid_rmse = session.run(
        models['rmse'],
        feed_dict={
            models['u']: batch_manager.valid_data[:, 0],
            models['i']: batch_manager.valid_data[:, 1],
            models['r']: batch_manager.valid_data[:, 2]
        })

    test_rmse = session.run(
        models['rmse'],
        feed_dict={
            models['u']: batch_manager.test_data[:, 0],
            models['i']: batch_manager.test_data[:, 1],
            models['r']: batch_manager.test_data[:, 2]
        })

    return valid_rmse, test_rmse


def get_p_and_q(kind, use_cache=True):
    if use_cache:
        try:
            p = np.load('llorma_p/{}-p.npy'.format(kind))
            q = np.load('llorma_p/{}-q.npy'.format(kind))
            return p, q
        except:
            print('>> There is no cached p and q.')

    batch_manager = BatchManager(kind)
    models = init_models_for_pre_train(batch_manager)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=GPU_MEMORY_FRAC)
    gpu_config = tf.ConfigProto(gpu_options=gpu_options)

    session = tf.Session(config=gpu_config)
    session.run(tf.global_variables_initializer())

    min_valid_rmse = float("Inf")
    min_valid_iter = 0
    final_test_rmse = float("Inf")

    random_model_idx = random.randint(0, 1000000)

    file_path = "tmp/model-{}.ckpt".format(random_model_idx)

    u = batch_manager.train_data[:, 0]
    i = batch_manager.train_data[:, 1]
    r = batch_manager.train_data[:, 2]

    saver = tf.train.Saver()
    for iter in range(1000000):
        for train_op in models['train_ops']:
            _, loss, train_rmse = session.run(
                (train_op, models['loss'], models['rmse']),
                feed_dict={models['u']: u,
                           models['i']: i,
                           models['r']: r})

        valid_rmse, test_rmse = _validate(session, batch_manager, models)

        if valid_rmse < min_valid_rmse:
            min_valid_rmse = valid_rmse
            min_valid_iter = iter
            final_test_rmse = test_rmse
            saver.save(session, file_path)

        if iter >= min_valid_iter + 100:
            break

        print('>> ITER:',
              "{:3d}".format(iter), "{:3f}, {:3f} {:3f} / {:3f}".format(
                  train_rmse, valid_rmse, test_rmse, final_test_rmse))

    saver.restore(session, file_path)
    p, q = session.run(
        (models['p'], models['q']),
        feed_dict={
            models['u']: batch_manager.train_data[:, 0],
            models['i']: batch_manager.train_data[:, 1],
            models['r']: batch_manager.train_data[:, 2]
        })
    np.save('llorma_p/{}-p.npy'.format(kind), p)
    np.save('llorma_p/{}-q.npy'.format(kind), q)

    session.close()
    return p, q
