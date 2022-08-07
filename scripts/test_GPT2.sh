#!/bin/bash
use_tcp="false"
tcp_path="None"
#tcp_path="outputs/TCP/best_model_test.txt"

test_path="data/sample_test.json"
cache_dir="caches/GPT2"
model_dir="logs/GPT2"
output_dir="outputs/GPT2"
max_length=80

python backbones/GPT-2/run_infer.py --use_tcp ${use_tcp} \
        --tcp_path ${tcp_path} \
        --test_path ${test_path} \
        --cache_dir ${cache_dir} \
        --model_dir ${model_dir} \
        --output_dir ${output_dir} \
        --max_length ${max_length}