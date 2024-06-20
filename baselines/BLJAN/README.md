## BLJAN

This is an implementation for "Byte-label joint attention learning 
for packet-grained network traffic classification".

#### Run

The data preprocessing & Training & evaluation phases of `BLJAN `
is roughly the same as `RBLJAN`.

* for preprocessing, the main difference is that `BLJAN` did not separate the header and the payload, you can modify data_preprocessing.py in `RBLJAN`.
* the training method is the same as `RBLJAN` without GAN, just `train.py -> train_BLJAN()`.
