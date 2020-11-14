# BiTraP: Bi-directional Pedestrian Trajectory Prediction with Multi-modal Goal Estimation
Yu Yao, Ella Atkins, Matthew Johnson-Roberson, Ram Vasudevan and Xiaoxiao Du


## Training

### Bounding box trajectory prediction on JAAD and PIE
Test BiTraP-NP on PIE
`python tools/test.py --config_file configs/gaussian_NP_PIE.yml CKPT_DIR **DIR_TO_CKPT**`

Test BiTraP-GMM on PIE
`python tools/test.py --config_file configs/Cat_GMM_PIE.yml CKPT_DIR **DIR_TO_CKPT**`

Test BiTraP-NP on JAAD
`python tools/test.py --config_file configs/gaussian_NP_JAAD.yml CKPT_DIR **DIR_TO_CKPT**`

Test BiTraP-GMM on JAAD
`python tools/test.py --config_file configs/Cat_GMM_JAAD.yml CKPT_DIR **DIR_TO_CKPT**`

### Point trajectory prediction on ETH-UCY
Test BiTraP-NP on JAAD
`python tools/test.py --config_file configs/gaussian_NP_ETH.yml DATASET.NAME **NAME_OF_DATASET** CKPT_DIR **DIR_TO_CKPT**`

Test BiTraP-GMM on JAAD
`python tools/test.py --config_file configs/Cat_GMM_ETH.yml DATASET.NAME **NAME_OF_DATASET** CKPT_DIR **DIR_TO_CKPT**`