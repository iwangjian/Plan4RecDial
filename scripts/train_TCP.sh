#!/bin/bash

# dataset
train_data="data/sample_train.json"
dev_data="data/sample_dev.json"
bert_dir="config/bert-base-chinese"
use_knowledge_hop="false"
cache_dir="caches/TCP"
log_dir="logs/TCP"

# train args
num_epochs=10
batch_size=6

python main.py --mode train \
    --train_data ${train_data} \
    --dev_data ${dev_data} \
    --bert_dir ${bert_dir} \
    --use_knowledge_hop ${use_knowledge_hop} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --num_epochs ${num_epochs} \
    --batch_size ${batch_size}