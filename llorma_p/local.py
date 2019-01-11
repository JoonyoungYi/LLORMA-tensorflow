import time
import math

import numpy as np

from .configs import *


def _create_p_or_q(n, rank, batch_manager):
    mu = batch_manager.mu
    std = batch_manager.std

    _mu = math.sqrt(mu / rank)
    _std = math.sqrt((math.sqrt(mu * mu + std * std) - mu) / rank)
    return np.random.normal(_mu, _std, [n, rank])


def _assign_p_and_q(session, models, p, q):
    p_assign_op = models['local_p'].assign(p)
    q_assign_op = models['local_q'].assign(q)

    session.run((p_assign_op, q_assign_op))


class LocalModel:
    def __init__(self, session, models, anchor_idx, anchor_manager,
                 batch_manager):
        self.session = session
        self.models = models
        self.batch_manager = batch_manager
        self.anchor_idx = anchor_idx
        self.anchor_manager = anchor_manager
        self.p = _create_p_or_q(batch_manager.n_user, LOCAL_RANK,
                                batch_manager)
        self.q = _create_p_or_q(batch_manager.n_item, LOCAL_RANK,
                                batch_manager)

        print('>> update k in anchor_idx [{}].'.format(anchor_idx))
        self.train_k = anchor_manager.get_train_k(anchor_idx)
        self.test_k = anchor_manager.get_test_k(anchor_idx)

        _assign_p_and_q(session, models, self.p, self.q)
        self._update_r_hats()

    def _update_r_hats(self):
        session = self.session
        models = self.models
        batch_manager = self.batch_manager

        train_r_hat = session.run(
            models['local_r_hat'],
            feed_dict={
                models['u']: batch_manager.train_data[:, 0],
                models['i']: batch_manager.train_data[:, 1],
                models['r']: batch_manager.train_data[:, 2],
            })

        test_r_hat = session.run(
            models['local_r_hat'],
            feed_dict={
                models['u']: batch_manager.test_data[:, 0],
                models['i']: batch_manager.test_data[:, 1],
                models['r']: batch_manager.test_data[:, 2],
            })

        self.train_r_hat = train_r_hat
        self.test_r_hat = test_r_hat

    def train(self):
        session = self.session
        models = self.models
        batch_manager = self.batch_manager
        anchor_idx = self.anchor_idx
        anchor_manager = self.anchor_manager
        p = self.p
        q = self.q
        train_k = self.train_k

        _assign_p_and_q(session, models, p, q)

        train_data = batch_manager.train_data
        prev_train_rmse = 5.0
        sum_batch_sse = 0.0
        n_batch = 0
        for iter in range(1, 100 + 1):
            for m in range(0, train_data.shape[0], BATCH_SIZE):
                end_m = min(m + BATCH_SIZE, train_data.shape[0])
                u = train_data[m:end_m, 0]
                i = train_data[m:end_m, 1]
                r = train_data[m:end_m, 2]
                k = train_k[m:end_m]
                sse, _ = session.run(
                    (models['local_sse'], models['local_train_op']),
                    feed_dict={
                        models['u']: u,
                        models['i']: i,
                        models['r']: r,
                        models['k']: k,
                    })
                sum_batch_sse += sse
                n_batch += u.shape[0]

            train_rmse = math.sqrt(sum_batch_sse / n_batch)
            if iter % 10 == 0:
                print('  - ITER [{:3d}]'.format(iter), train_rmse)

            if abs(prev_train_rmse - train_rmse) < 1e-4:
                break
            prev_train_rmse = train_rmse
            p, q = session.run((models['local_p'], models['local_q']))
            sum_batch_sse, n_batch = 0, 0

        self.p, self.q = p, q
        _assign_p_and_q(session, models, p, q)

        self._update_r_hats()

        self.train_k = train_k
