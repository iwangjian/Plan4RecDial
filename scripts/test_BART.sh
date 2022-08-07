#!/bin/bash
use_tcp="false"
tcp_path="None"
#tcp_path="outputs/TCP/best_model_test.txt"

test_path="data/sample_test.json"
cache_dir="caches/BART"
model_dir="logs/BART"
output_dir="outputs/BART"
test_batch_size=4
max_dec_len=80

python backbones/BART/run_infer.py --use_tcp ${use_tcp} \
        --tcp_path ${tcp_path} \
        --test_path ${test_path} \
        --cache_dir ${cache_dir} \
        --model_dir ${model_dir} \
        --output_dir ${output_dir} \
        --test_batch_size ${test_batch_size} \
        --max_dec_len ${max_dec_len}