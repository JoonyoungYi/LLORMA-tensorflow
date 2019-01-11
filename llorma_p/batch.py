import numpy as np

from base.dataset import DatasetManager
from .configs import *


class BatchManager:
    def __init__(self, kind):
        self.kind = kind
        dataset_manager = DatasetManager(kind, N_SHOT)
        self.train_data = np.concatenate(
            [
                dataset_manager.get_train_data(),
                dataset_manager.get_valid_data()
            ],
            axis=0)
        self.test_data = dataset_manager.get_test_data()

        self.n_user = int(
            max(np.max(self.train_data[:, 0]), np.max(self.test_data[:,
                                                                     0]))) + 1
        self.n_item = int(
            max(np.max(self.train_data[:, 1]), np.max(self.test_data[:,
                                                                     1]))) + 1
        self.mu = np.mean(self.train_data[:, 2])
        self.std = np.std(self.train_data[:, 2])
