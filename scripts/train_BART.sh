#!/bin/bash
use_tcp="false"
train_path="data/sample_train.json"
valid_path="data/sample_dev.json"
cache_dir="caches/BART"
log_dir="logs/BART"

n_epochs=5
lr=2e-5
warmup_steps=3000
train_batch_size=16
valid_batch_size=16

python backbones/BART/run_train.py --use_tcp ${use_tcp} \
        --train_path ${train_path} \
        --valid_path ${valid_path} \
        --cache_dir ${cache_dir} \
        --log_dir ${log_dir} \
        --n_epochs ${n_epochs} \
        --lr ${lr} \
        --warmup_steps ${warmup_steps} \
        --train_batch_size ${train_batch_size} \
        --valid_batch_size ${valid_batch_size}