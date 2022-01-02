# TCP-RecDial
Follow Me: Target-driven Conversation Planning for Proactive Recommendation Dialogue Systems


## Requirements
This project is implemented based on Python 3.x. To install the dependencies, please run:
```
pip install -r requirements.txt
```
Note: (1) The pretrained Chinese BERT model can be downloaded from Hugging Face's [model card](https://huggingface.co/bert-base-chinese/tree/main), please download `pytorch_model.bin` and place the model file into the `config/bert-base-chinese/` folder. 

(2) For fine-tuning CDial-GPT, please download the pretrained model `pytorch_model.bin` from [here](https://huggingface.co/thu-coai/CDial-GPT_LCCC-base/tree/main) and place the model file into the `config/CDial-GPT_LCCC-base/` folder. For fine-tuning Chinese version of GPT-2, please download the pretrained model `pytorch_model.bin` from [here](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall/tree/main) and place the model file into the `config/gpt2-chinese-cluecorpussmall/` folder. For fine-tuning ERNIE-GEN, please download the pretrained ERNIE 1.0 model from [here](https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz) and unzip all files into the `config/ERNIE_1.0_max-len-512/` folder.

(3) The backbone model ERNIE-GEN is based on [PaddlePaddle](https://www.paddlepaddle.org.cn/) framework, please follow the `backbones/ERNIE-GEN/requirements.txt` to install the dependencies.


## Dataset
We have uploaded the preprocessed DuRecDial dataset to the `data/` folder, where the `.zip` files need to be unzipped. For more details about the original DuRecDial dataset, please refer to [here](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2020-DuRecDial).

## Quickstart

### Conversation Planning

For training the TCP model, please refer to the `main.py` and set parameters in the `train_TCP.sh` accordingly, then run:
```
sh train_TCP.sh
```

For generating plans using the TCP model, please refer to the `main.py` and set parameters in the `test_TCP.sh` accordingly, then run:
```
sh test_TCP.sh
```
Note: If you additionally adopt set-search decoding (`--use_ssd=True`), the parameter `test_batch_size` shoud be set to `1`.


### Dialogue Generation

Quick running scripts of dialogue training include `train_CDial-GPT.sh`, `train_GPT2.sh`, `preprocess_ERNIE-GEN.sh` and `train_ERNIE-GEN.sh`.
For training baselines, please ensure `use_tcp="false"`, while for training TCP-enhanced dialogue models, please set `use_tcp="true"`. Then, please run:
```
# CDial-GPT
sh train_CDial-GPT.sh

# GPT-2
sh train_GPT2.sh

# ERNIE-GEN
sh preprocess_ERNIE-GEN.sh
sh train_ERNIE-GEN.sh
```

For dialogue generation, quick running scripts include `test_CDial-GPT.sh`, `test_GPT2.sh`, and `test_ERNIE-GEN.sh`, please first set `use_tcp="false"` or `use_tcp="true"`. If `use_tcp="true"`, `tcp_path` should be specified as the output path of generated plans. Then, please run:
```
# CDial-GPT
sh test_CDial-GPT.sh

# GPT-2
sh test_GPT2.sh

# ERNIE-GEN
sh test_ERNIE-GEN.sh
```

### Evaluation
For evaluation of conversation planning, please run:
```
python eval/eval_planning.py --eval_file ${eval_file} --gold_file data/sample_test.json
```

For evaluation of dialogue generation, please run:
```
python eval/eval_generation.py --eval_file ${eval_file} --gold_file data/sample_test.json
```
Note: `${eval_file}` should be replaced by the actual file path to be evaluated.
