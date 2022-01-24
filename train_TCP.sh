#!/bin/bash

# dataset
train_data="data/sample_train.json"
dev_data="data/sample_dev.json"
test_data="data/sample_test.json"
bert_dir="config/bert-base-chinese"
cache_dir="caches/TCP"

# train args
log_dir="logs/TCP"
num_epochs=10
batch_size=4

python3 main.py --mode train \
    --train_data ${train_data} \
    --dev_data ${dev_data} \
    --test_data ${test_data} \
    --bert_dir ${bert_dir} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --num_epochs ${num_epochs} \
    --batch_size ${batch_size}