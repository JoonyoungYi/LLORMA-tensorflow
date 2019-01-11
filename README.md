# LLORMA-tensorflow

* This repository is tensorflow implementation of "Local Low-rank Matrix Approximation".
  * I implemented two version of LLORMA: [Parallel LLORMA (ICML'13)](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45235.pdf) and [Global LLORMA (JMLR'16)](http://jmlr.org/papers/volume17/14-301/14-301.pdf).
* I have increased the batch size for performance. If you want to get results as same as the original paper, please set batch size to 1.
* I refer the codes from https://github.com/jnhwkim/PREA/tree/master/src/main/java/prea/recommender/llorma.

* Folder description
  * llorma_p: Parallel LLORMA (ICML'13)
  * llorma_g: Global LLORMA (JMLR'16)

* Dependecy: `python3.5` and see `requirements.txt`

* How to run
```
python train.py
```
