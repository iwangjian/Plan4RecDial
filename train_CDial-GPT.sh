#!/bin/bash
use_tcp="true"
train_path="data/sample_train.json"
valid_path="data/sample_dev.json"
cache_dir="caches/CDial-GPT"

model_checkpoint="config/CDial-GPT_LCCC-base"
log_dir="logs/CDial-GPT"
n_epochs=10
lr=5e-5
warmup_steps=3000
train_batch_size=16
valid_batch_size=16

python3 backbones/CDial-GPT/train.py --use_tcp ${use_tcp} \
        --train_path ${train_path} \
        --valid_path ${valid_path} \
        --cache_dir ${cache_dir} \
        --model_checkpoint ${model_checkpoint} \
        --log_dir ${log_dir} \
        --n_epochs ${n_epochs} \
        --lr ${lr} \
        --warmup_steps ${warmup_steps} \
        --train_batch_size ${train_batch_size} \
        --valid_batch_size ${valid_batch_size}