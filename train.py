from base.dataset import DatasetManager

from llorma_p.trainer import main as llorma_parallel_train
from llorma_g.trainer import main as llorma_global_train

if __name__ == '__main__':
    # kind = DatasetManager.KIND_MOVIELENS_100K
    kind = DatasetManager.KIND_MOVIELENS_1M
    # kind = DatasetManager.KIND_MOVIELENS_10M
    # kind = DatasetManager.KIND_MOVIELENS_20M

    # llorma_parallel_train(kind)
    llorma_global_train(kind)
