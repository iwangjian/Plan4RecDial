#!/bin/bash

# dataset
test_data="data/sample_test.json"
bert_dir="config/bert-base-chinese"
cache_dir="caches/TCP"

# decode args
model_dir="logs/TCP"
output_dir="outputs/TCP"
use_ssd="True"
test_batch_size=1

python3 main.py --mode test \
    --test_data ${test_data} \
    --bert_dir ${bert_dir} \
    --cache_dir ${cache_dir} \
    --model_dir ${model_dir} \
    --output_dir ${output_dir} \
    --use_ssd ${use_ssd} \
    --test_batch_size ${test_batch_size}