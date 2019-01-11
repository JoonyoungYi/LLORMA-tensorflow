import time
import math

import numpy as np

from .configs import *


class LocalModel:
    def __init__(self, session, models, anchor_idx, anchor_manager,
                 batch_manager):
        self.session = session
        self.models = models
        self.batch_manager = batch_manager
        self.anchor_idx = anchor_idx
        self.anchor_manager = anchor_manager

        print('>> update k in anchor_idx [{}].'.format(anchor_idx))
        self.train_k = anchor_manager.get_train_k(anchor_idx)
        self.valid_k = anchor_manager.get_valid_k(anchor_idx)
        self.test_k = anchor_manager.get_test_k(anchor_idx)
