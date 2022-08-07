#!/bin/bash
use_tcp="false"
tcp_path="None"
#tcp_path="outputs/TCP/best_model_test.txt"

test_path="data/sample_test.json"
cache_dir="caches/CDial-GPT"
model_dir="logs/CDial-GPT"
output_dir="outputs/CDial-GPT"
max_length=80

python backbones/CDial-GPT/run_infer.py --use_tcp ${use_tcp} \
        --tcp_path ${tcp_path} \
        --test_path ${test_path} \
        --cache_dir ${cache_dir} \
        --model_dir ${model_dir} \
        --output_dir ${output_dir} \
        --max_length ${max_length}