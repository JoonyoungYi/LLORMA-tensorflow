import os
import random

import numpy as np

# from ..configs import *


def _make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


class DatasetManager:
    KIND_MOVIELENS_100K = 'movielens-100k'
    KIND_MOVIELENS_1M = 'movielens-1m'
    KIND_MOVIELENS_10M = 'movielens-10m'
    KIND_MOVIELENS_20M = 'movielens-20m'
    KIND_NETFLIX = 'netflix'

    KIND_OBJECTS = ( \
        (KIND_MOVIELENS_100K, 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'), \
        (KIND_MOVIELENS_1M,  'http://files.grouplens.org/datasets/movielens/ml-1m.zip'), \
        (KIND_MOVIELENS_10M, 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'), \
        (KIND_MOVIELENS_20M, 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'), \
        (KIND_NETFLIX, None)
    )

    def _set_kind_and_url(self, kind):
        self.kind = kind
        for k, url in self.KIND_OBJECTS:
            if k == kind:
                self.url = url
                return True
        raise NotImplementedError()

    def _download_data_if_not_exists(self):
        if not os.path.exists('data/{}'.format(self.kind)):
            os.system('wget {url} -O data/{kind}.zip'.format(
                url=self.url, kind=self.kind))
            os.system(
                'unzip data/{kind}.zip -d data/{kind}/'.format(kind=self.kind))

    def __init_data(self, detail_path, delimiter, header=False):
        current_u = 0
        u_dict = {}
        current_i = 0
        i_dict = {}

        data = []
        with open('data/{}{}'.format(self.kind, detail_path), 'r') as f:
            if header:
                f.readline()

            for line in f:
                cols = line.strip().split(delimiter)
                assert len(cols) == 4
                # cols = [float(c) for c in cols]
                user_id = cols[0]
                item_id = cols[1]
                r = float(cols[2])
                t = int(cols[3])

                u = u_dict.get(user_id, None)
                if u is None:
                    u_dict[user_id] = current_u
                    u = current_u
                    current_u += 1

                i = i_dict.get(item_id, None)
                if i is None:
                    # print(current_i)
                    i_dict[item_id] = current_i
                    i = current_i
                    current_i += 1

                data.append((u, i, r, t))
            f.close()

        data = np.array(data)
        np.save('data/{}/data.npy'.format(self.kind), data)

    def _init_data(self):
        if self.kind == self.KIND_MOVIELENS_100K:
            self.__init_data('/ml-100k/u.data', '\t')
        elif self.kind == self.KIND_MOVIELENS_1M:
            self.__init_data('/ml-1m/ratings.dat', '::')
        elif self.kind == self.KIND_MOVIELENS_10M:
            self.__init_data('/ml-10M100K/ratings.dat', '::')
        elif self.kind == self.KIND_MOVIELENS_20M:
            self.__init_data('/ml-20m/ratings.csv', ',', header=True)
        else:
            raise NotImplementedError()

    def _load_base_data(self):
        return np.load('data/{}/data.npy'.format(self.kind))

    def _split_data(self):
        data = self.data
        n_shot = self.n_shot
        np.random.shuffle(data)

        if self.n_shot == -1:
            # n_shot이 -1일때는 더 sparse하게 전체 레이팅을 9:1로 test train set을 나눈다.
            n_train = int(data.shape[0] * 0.1)
            n_valid = int(n_train * 0.9)

            train_data = data[:n_valid]
            valid_data = data[n_valid:n_train]
            test_data = data[n_train:]

            np.save(self._get_npy_path('train'), train_data)
            np.save(self._get_npy_path('valid'), valid_data)
            np.save(self._get_npy_path('test'), test_data)

        elif self.n_shot == 0:
            # n_shot이 0일때는 다른 알고리즘들처럼 전체 레이팅을 1:9로 test train set을 나눈다.
            n_train = int(data.shape[0] * 0.9)
            n_valid = int(n_train * 0.98)

            train_data = data[:n_valid]
            valid_data = data[n_valid:n_train]
            test_data = data[n_train:]

            np.save(self._get_npy_path('train'), train_data)
            np.save(self._get_npy_path('valid'), valid_data)
            np.save(self._get_npy_path('test'), test_data)

        else:
            # 전체 유저 중에 20%를 일단 test user로 뗍니다.
            test_user_ids = random.sample(
                list(range(self.n_user)), self.n_user // 5)

            train_data = []
            test_data = []
            count_dict = {}
            for i in range(data.shape[0]):
                row = data[i]
                user_id = int(row[0])
                if user_id in test_user_ids:
                    count = count_dict.get(user_id, 0)
                    if count < n_shot:
                        train_data.append(row)
                    else:
                        test_data.append(row)
                    count_dict[user_id] = count + 1
                else:
                    train_data.append(row)

            train_data = np.array(train_data)
            n_valid = int(train_data.shape[0] * 0.98)
            train_data, valid_data = train_data[:n_valid], train_data[n_valid:]

            np.save(self._get_npy_path('train'), train_data)
            np.save(self._get_npy_path('valid'), valid_data)

            test_data = np.array(test_data)
            np.save(self._get_npy_path('test'), test_data)

    def _get_npy_path(self, split_kind):
        return 'data/{}/shot-{}/{}.npy'.format(self.kind, self.n_shot,
                                               split_kind)

    def __init__(self, kind, n_shot=0):
        assert type(n_shot) == int and n_shot >= -1

        _make_dir_if_not_exists('data')
        self._set_kind_and_url(kind)
        self._download_data_if_not_exists()
        self.n_shot = n_shot

        # 예쁜 형태로 정제된 npy 파일이 없으면, 정제를 수행합니다.
        if not os.path.exists('data/{}/data.npy'.format(kind)):
            self._init_data()
        self.data = self._load_base_data()

        _make_dir_if_not_exists(
            'data/{}/shot-{}'.format(self.kind, self.n_shot))

        self.n_user = int(np.max(self.data[:, 0])) + 1
        self.n_item = int(np.max(self.data[:, 1])) + 1
        self.n_row = self.n_user
        self.n_col = self.n_item

        # split된 데이터가 없으면 split합니다.
        if not os.path.exists(
                self._get_npy_path('train')) or not os.path.exists(
                    self._get_npy_path('valid')) or not os.path.exists(
                        self._get_npy_path('test')):
            self._split_data()

        self.train_data = np.load(self._get_npy_path('train'))
        self.valid_data = np.load(self._get_npy_path('valid'))
        self.test_data = np.load(self._get_npy_path('test'))

    def get_train_data(self):
        return self.train_data

    def get_valid_data(self):
        return self.valid_data

    def get_test_data(self):
        return self.test_data


# if __name__ == '__main__':
#     kind = DatasetManager.KIND_MOVIELENS_100K
#     kind = DatasetManager.KIND_MOVIELENS_1M
#     kind = DatasetManager.KIND_MOVIELENS_10M
#     kind = DatasetManager.KIND_MOVIELENS_20M
#     dataset_manager = DatasetManager(kind)
