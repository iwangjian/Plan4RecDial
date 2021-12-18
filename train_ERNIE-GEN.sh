#!/bin/bash
export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES="0"
export PADDLE_TRAINERS_NUM=1

# set TCP
use_tcp="true"     # use_tcp= "true" or "flase"

# load pretrained ERNIE-1.0 model
init_model="config/ERNIE_1.0_max-len-512/params"
vocab_path="config/ERNIE_1.0_max-len-512/vocab.txt"
config_path="config/ERNIE_1.0_max-len-512/ernie_config.json"

# dataset
data_path="caches/ERNIE-GEN"
train_set="train.tsv"
dev_set="valid.tsv"

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

# train
checkpoints="logs/ERNIE-GEN"
epoch=10
weight_decay=0.01
label_smooth=0.1
hidden_dropout_prob=0.1
save_and_valid_by_epoch="True"
save_steps=10000
validation_steps=10000

# lr
warmup_proportion=0.1
lr_scheduler="linear_warmup_decay"
learning_rate=1e-4
random_noise="false"
noise_prob=0.0
random_seed=42

if [ "${use_tcp}" = "true" ] || [ "${use_tcp}" = "True" ]; then
    data_path="${data_path}_w_TCP"
    checkpoints="${checkpoints}_w_TCP"
    echo ${data_path}
    echo ${checkpoints}
fi

mkdir -p ${checkpoints}

python3 backbones/ERNIE-GEN/run_seq2seq.py --use_cuda true \
    --do_train true \
    --do_val false \
    --do_test false \
    --do_pred false \
    --train_set ${data_path}/${train_set:-""} \
    --dev_set ${data_path}/${dev_set:-""} \
    --task_type ${task_type:-"dialog"} \
    --role_type_size ${role_type_size:-0} \
    --turn_type_size ${turn_type_size:-0} \
    --max_src_len ${max_src_len} \
    --max_tgt_len ${max_tgt_len} \
    --tokenizer ${tokenizer:-"FullTokenizer"} \
    --tokenized_input ${tokenized_input:-"False"} \
    --continuous_position ${continuous_position:-"false"} \
    --batch_size ${batch_size} \
    --init_pretraining_params ${init_model:-""} \
    --vocab_path ${vocab_path} \
    --ernie_config_path ${config_path} \
    --checkpoints ${checkpoints} \
    --epoch ${epoch} \
    --weight_decay ${weight_decay:-0.0} \
    --weight_sharing ${weight_sharing:-"True"} \
    --label_smooth ${label_smooth:-0.0} \
    --hidden_dropout_prob ${hidden_dropout_prob:--1} \
    --attention_probs_dropout_prob ${attention_probs_dropout_prob:--1} \
    --save_and_valid_by_epoch ${save_and_valid_by_epoch} \
    --save_steps ${save_steps} \
    --validation_steps ${validation_steps} \
    --warmup_proportion ${warmup_proportion:-0.0} \
    --lr_scheduler ${lr_scheduler:-"linear_warmup_decay"} \
    --learning_rate ${learning_rate} \
    --random_noise ${random_noise:-"False"} \
    --noise_prob ${noise_prob:-0.0} \
    --random_seed ${random_seed}
