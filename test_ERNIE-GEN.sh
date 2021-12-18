#!/bin/bash
export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES="0"
export PADDLE_TRAINERS_NUM=1

# set TCP
use_tcp="true"     # use_tcp= "true" or "flase"

# set log dir and checkpoint epoch
log_dir="logs/ERNIE-GEN"
checkpoint="epoch_9"

# load config
vocab_path="config/ERNIE_1.0_max-len-512/vocab.txt"
config_path="config/ERNIE_1.0_max-len-512/ernie_config.json"

# dataset
data_path="caches/ERNIE-GEN"
pred_set="test.tsv"

# input
task_type="dialog"
role_type_size=3
turn_type_size=16
max_src_len=430
max_tgt_len=82
tokenizer="FullTokenizer"
tokenized_input="false"
continuous_position="true"
batch_size=16

# decode
do_decode="true"
output_dir="outputs/ERNIE-GEN"
max_dec_len=82
beam_size=1
length_penalty=1.3
random_seed=42

if [ "${use_tcp}" = "true" ] || [ "${use_tcp}" = "True" ]; then
    init_model="${log_dir}_w_TCP/${checkpoint}"
    data_path="${data_path}_w_TCP"
    output_dir="${output_dir}_w_TCP"
else
    init_model=${log_dir}/${checkpoint}
fi
echo ${init_model}
echo ${data_path}
echo ${output_dir}

python3 backbones/ERNIE-GEN/run_seq2seq.py --use_cuda true \
    --do_train fasle \
    --do_val false \
    --do_test false \
    --do_pred true \
    --pred_set ${data_path}/${pred_set:-""} \
    --task_type ${task_type:-"dialog"} \
    --role_type_size ${role_type_size:-0} \
    --turn_type_size ${turn_type_size:-0} \
    --max_src_len ${max_src_len} \
    --max_tgt_len ${max_tgt_len} \
    --max_dec_len ${max_dec_len} \
    --tokenizer ${tokenizer:-"FullTokenizer"} \
    --tokenized_input ${tokenized_input:-"False"} \
    --continuous_position ${continuous_position:-"false"} \
    --batch_size ${batch_size} \
    --do_decode ${do_decode:-"True"} \
    --output_dir ${output_dir} \
    --beam_size ${beam_size:-5}  \
    --length_penalty ${length_penalty:-"0.0"} \
    --init_pretraining_params ${init_model:-""} \
    --vocab_path ${vocab_path} \
    --ernie_config_path ${config_path} \
    --random_seed ${random_seed}
