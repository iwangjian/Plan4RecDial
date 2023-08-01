# Plan4RecDial
This repo serves as the codebase for "[Follow Me: Conversation Planning for Target-driven Recommendation Dialogue Systems](https://arxiv.org/abs/2208.03516)" (arXiv 2022) and the extended version "[A Target-Driven Planning Approach for Goal-Directed Dialog Systems](https://ieeexplore.ieee.org/document/10042178/)" (TNNLS 2023).

We move forward to a promising yet under-explored proactive dialogue paradigm called "target-driven recommendation dialogue systems" (or "goal-directed dialogue systems", where the "goal" refers to leading conversations to the designated target and achieving recommendations). We focus on how to equip such a dialogue system with the ability to naturally lead users to reach the target through reasonable action transitions and smooth topic transitions. To this end, we propose a target-driven planning framework, which plans (generates) a dialogue path consisting of a sequence of dialogue action-topic pairs, driving the system to proactively transit between different conversation stages. We then apply the planned dialogue path to guide dialogue generation using various backbone models in a pipeline manner.


## Requirements
This project is implemented based on Python 3. To install the dependencies, please run:
```
pip install -r requirements.txt
```
Note: (1) The pretrained Chinese BERT model can be downloaded from Hugging Face's [model card](https://huggingface.co/bert-base-chinese/tree/main), please download `pytorch_model.bin` and place the model file into the `config/bert-base-chinese/` folder. 

(2) For fine-tuning CDial-GPT, please download the pretrained model `pytorch_model.bin` from [here](https://huggingface.co/thu-coai/CDial-GPT_LCCC-base/tree/main) and place the model file into the `config/CDial-GPT_LCCC-base/` folder. For fine-tuning BART, please download the Chinese version pretrained model from [here](https://huggingface.co/fnlp/bart-base-chinese/tree/main) and place the model file into the `config/bart-base-chinese` folder. For fine-tuning GPT-2, please download the  Chinese version pretrained model `pytorch_model.bin` from [here](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall/tree/main) and place the model file into the `config/gpt2-chinese-cluecorpussmall/` folder.


## Dataset and Processing
We describe how we re-purpose the DuRecDial dataset in the extended version paper, and we have uploaded the re-purposed dataset to the `data/` folder. All `sample_*.json` files are used in the experiments, where necessary key-value pairs in a JSON line are illustrated as follows:
```
{
    "user_profile":  // User profile in the form of key-value pairs
    "knowledge":  // Domain knowledge in the form of <s, r, o> triples
    "target":  // Designated target action and target topic for the whole conversation
    "conversation":  // Dialogue history consisting of user-system utterances
    "action_path":  // Dialogue action path from the current turn to the end turn of the conversation
    "topic_path":  // Dialogue topic path from the current turn to the end turn of the conversation
    "response":  // System utterance at the current turn
}
```
For more details about the original DuRecDial dataset, please refer to [here](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2020-DuRecDial).

## Quickstart

### Conversation Planning

For training the TCP model, please refer to the `main.py` and set parameters in the `scripts/train_TCP.sh` accordingly, then run:
```
sh scripts/train_TCP.sh
```

For plan path generation, please refer to the `main.py` and set parameters in the `scripts/test_TCP.sh` accordingly, then run:
```
sh scripts/test_TCP.sh
```
Note: If adopt the set-search decoding strategy (`--use_ssd="true"`), the parameter `test_batch_size` shoud be set to `1`.


### Dialogue Generation

Quickstart scripts for dialogue training are included under the folder `scripts/`.
Note that for training baseline models, please set `use_tcp="false"`, while for training TCP-enhanced dialogue models, please set `use_tcp="true"` accordingly. Then, please run:
```
# CDial-GPT
sh scripts/train_CDial-GPT.sh

# BART
sh scripts/train_BART.sh

# GPT-2
sh scripts/train_GPT2.sh
```

For dialogue generation, quickstart scripts are also included under the folder `scripts/`. For baseline models, please set `use_tcp="false"`. For TCP-enhanced dialogue generation, please set `use_tcp="true"` and `tcp_path=${TCP_PATH}` (Note that `${TCP_PATH}` should be specified as the actual file path of the generated plans by TCP). Then, please run:
```
# CDial-GPT
sh scripts/test_CDial-GPT.sh

# BART
sh scripts/test_BART.sh

# GPT-2
sh scripts/test_GPT2.sh
```

### Evaluation
For evaluation of conversation planning, please run:
```
python eval/eval_planning.py --eval_file ${eval_file} --gold_file data/sample_test.json
```

For evaluation of dialogue generation, please run:
```
python eval/eval_dialogue.py --eval_file ${eval_file} --gold_file data/sample_test.json
```
Note: `${eval_file}` should be specified as the actual file path to be evaluated.


### Citation
If you find this repo helpful, please cite the following papers:
```bibtex
@article{wang2022follow,
  title={Follow Me: Conversation Planning for Target-driven Recommendation Dialogue Systems},
  author={Wang, Jian and Lin, Dongding and Li, Wenjie},
  journal={arXiv preprint arXiv:2208.03516},
  year={2022}
}

@article{wang2023target,
  title={A Target-Driven Planning Approach for Goal-Directed Dialog Systems},
  author={Wang, Jian and Lin, Dongding and Li, Wenjie},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```