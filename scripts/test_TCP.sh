#!/bin/bash

# dataset
test_data="data/sample_test.json"
bert_dir="config/bert-base-chinese"
cache_dir="caches/TCP"
log_dir="logs/TCP"

# decode args
output_dir="outputs/TCP"
use_ssd="false"

python main.py --mode test \
    --test_data ${test_data} \
    --bert_dir ${bert_dir} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --output_dir ${output_dir} \
    --use_ssd ${use_ssd}