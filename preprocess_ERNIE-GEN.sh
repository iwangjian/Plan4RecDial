#!/bin/bash
use_tcp="true"
tcp_path="outputs/TCP/best_model_test.txt"
train_path="data/sample_train.json"
valid_path="data/sample_dev.json"
test_path="data/sample_test.json"
cache_dir="caches/ERNIE-GEN"


python3 backbones/ERNIE-GEN/preprocess.py --use_tcp ${use_tcp} \
        --tcp_path ${tcp_path} \
        --train_path ${train_path} \
        --valid_path ${valid_path} \
        --test_path ${test_path} \
        --cache_dir ${cache_dir}