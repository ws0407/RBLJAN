## EBSNN

This is an implementation for "EBSNN: extended byte segment neural network for network traffic classification". The original author's implementation is in this [link](https://github.com/DCMMC/EBSNN).

#### Run

The data preprocessing & Training & evaluation phases of `EBSNN `is roughly the same as `RBLJAN`.

* for preprocessing, the main difference is that `EBSNN` sets all port numbers to zero, and some fields may not be set to zero (such as `offset`) you can modify data_preprocessing.py according to the original paper.
* the training method is the same as RBLJAN without GAN, just `train.py -> train_EBSNN()`.
