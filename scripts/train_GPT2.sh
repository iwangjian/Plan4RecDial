#!/bin/bash
use_tcp="false"
train_path="data/sample_train.json"
valid_path="data/sample_dev.json"
cache_dir="caches/GPT2"

model_checkpoint="config/gpt2-chinese-cluecorpussmall"
log_dir="logs/GPT2"
n_epochs=5
lr=5e-5
warmup_steps=3000
train_batch_size=16
valid_batch_size=16

python backbones/GPT-2/run_train.py --use_tcp ${use_tcp} \
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