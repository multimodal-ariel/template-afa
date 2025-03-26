Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding
==

Python implementation of the model and learning algorithm proposed by Shim et
al., 2018

## Result

This is a deep learning method to solve multi-class classification problems in a
cost-sensitive setting. Given dataset whose feature acquisition costs are
defined, our model tries to find the optimal feature acquisition policy and
the classifier working with the policy. After training is done, this code prints
out the test results (classification performance, # of times that each feature is
acquired, etc.) of training/validation/test dataset.

<p align="center">
<img
src="https://github.com/OpenXAIProject/Joint-AFA-Classification/blob/master/dfs_result.png"  width="800">
</p>

## Dataset

CUBE dataset (used in the paper) is generated as default. You can also use your own dataset in the form of
the csv file whose first column is label followed by features values. Set
`data_type` argument as 'csv' and pass csv filename as a keyword argument named
`csv_filename` to `data_load` function. You should define
feature acquisition cost and pass it to `r_cost` argument.

## Installation

**1. Fork & Clone** : Fork this project to your repository and clone to your work directory.

  ```bash
  $ git clone https://github.com/OpenXAIProject/Joint-AFA-Classification.git
  ```

**2. Run** : Run python3 main.py --data_type=csv or cube_[feature
dimension]_[sigma]. You can check additional options by following command.

```bash
$ python3 main.py --help
usage: main.py [-h] [--disable_cuda] [--complete] [--pretrain PRETRAIN] [--pretrain_sample PRETRAIN_SAMPLE] [--mode MODE] [--scheduler SCHEDULER] [--dropout] [--batchnorm] [--done_action_train]
               [--data_type DATA_TYPE] [--p P] [--group_norm GROUP_NORM] [--save_dir SAVE_DIR] [--embedder_hidden_sizes EMBEDDER_HIDDEN_SIZES] [--clf_hidden_sizes CLF_HIDDEN_SIZES]
               [--policy_hidden_sizes POLICY_HIDDEN_SIZES] [--shared_dim SHARED_DIM] [--target_update_freq TARGET_UPDATE_FREQ] [--eps_start EPS_START] [--eps_end EPS_END]
               [--decay_rate DECAY_RATE] [--n_envs N_ENVS] [--nsteps NSTEPS] [--normalize NORMALIZE] [--embedded_dim EMBEDDED_DIM] [--lstm_size LSTM_SIZE] [--n_shuffle N_SHUFFLE]
               [--r_cost R_COST] [--cost_from_file COST_FROM_FILE] [--random_seed RANDOM_SEED] [--batch_size BATCH_SIZE] [--message MESSAGE]

options:
  -h, --help            show this help message and exit
  --disable_cuda        Disable CUDA
  --complete            train classifier with complete data
  --pretrain PRETRAIN   pre classifier training
  --pretrain_sample PRETRAIN_SAMPLE
  --mode MODE           double dqn?
  --scheduler SCHEDULER
                        ent_coef
  --dropout             Dropout classifier
  --batchnorm           batch norm
  --done_action_train   done action train
  --data_type DATA_TYPE
                        data
  --p P                 dropout prob
  --group_norm GROUP_NORM
                        group_norm regularization param
  --save_dir SAVE_DIR   save directory name
  --embedder_hidden_sizes EMBEDDER_HIDDEN_SIZES
                        embedder
  --clf_hidden_sizes CLF_HIDDEN_SIZES
                        clf mlp size
  --policy_hidden_sizes POLICY_HIDDEN_SIZES
                        a2c mlp size
  --shared_dim SHARED_DIM
                        a2c net shared vertor dim for pi and v
  --target_update_freq TARGET_UPDATE_FREQ
                        .
  --eps_start EPS_START
                        .
  --eps_end EPS_END     .
  --decay_rate DECAY_RATE
                        .
  --n_envs N_ENVS       how many episodes simultaneouly?
  --nsteps NSTEPS       num of steps for calc return
  --normalize NORMALIZE
                        make embedded feature l2 norm to 1
  --embedded_dim EMBEDDED_DIM
                        embedded vector dimension
  --lstm_size LSTM_SIZE
                        encoder lstm size
  --n_shuffle N_SHUFFLE
                        n shuffle
  --r_cost R_COST       cost weight(negative value)
  --cost_from_file COST_FROM_FILE
                        whether the cost info is in data csv file or not
  --random_seed RANDOM_SEED
                        random seed
  --batch_size BATCH_SIZE
                        batch size
  --message MESSAGE     message
```

## Requirements
+ python 3.5
+ pytorch (0.4.1)
+ numpy (1.15.0)
+ matplotlib (2.2.2)
+ scikit-learn (0.19.1)

## Reference
If you found the provided code useful, please cite our work.

```
@inproceedings{shim2018jointAFA,
    author    = {Hajin Shim and Sung Ju Hwangand Eunho Yang and },
    title     = {Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding},
    booktitle = {NIPS},
    year      = {2018}
              }
```

<br/>


## Contacts
If you have any question, please contact Hajin Shim(shimazing@kaist.ac.kr).

<br />
<br />

# XAI Project

**This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.2017-0-01779, A machine learning and statistical inference framework for explainable artificial intelligence)**

+ Project Name : A machine learning and statistical inference framework for explainable artificial intelligence(의사결정 이유를 설명할 수 있는 인간 수준의 학습·추론 프레임워크 개발)

+ Managed by Ministry of Science and ICT/XAIC <img align="right" src="http://xai.unist.ac.kr/static/img/logos/XAIC_logo.png" width=300px>

+ Participated Affiliation : UNIST, Korea Univ., Yonsei Univ., KAIST, AItrics

+ Web Site : <http://openXai.org>

