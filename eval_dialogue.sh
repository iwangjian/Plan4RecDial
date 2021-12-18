#!/bin/bash
python3 eval/eval_generation.py --eval_file=outputs/ERNIE-GEN_w_TCP/pred_output.txt \
    --gold_file=data/sample_test.json