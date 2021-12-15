# -*- coding: utf-8 -*-
import os
import json
from transformers import BertTokenizer

ACT = "[a]"
TOPIC = "[t]"
BOP = "[bop]"
EOP = "[eop]"
BOS = "[bos]"
EOS = "[eos]"

ATTR_TO_SPECIAL_TOKENS = {
    "additional_special_tokens": [ACT, TOPIC, BOP, EOP, BOS, EOS]
}


def get_tokenizer(config_dir):
    tokenizer = BertTokenizer.from_pretrained(config_dir)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKENS)
    return tokenizer, num_added_tokens

def load_data(file_path, is_test=False):
    data = []
    with open(file_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            sample = json.loads(line.strip())
            if is_test:
                data_dict = {
                    "user_profile": sample['user_profile'],
                    "knowledge_graph": sample['knowledge_graph'],
                    "hop": sample['hop'],
                    "conversation": sample['conversation'],
                    "target": ACT + sample['target'][0] + TOPIC + sample['target'][1]
            }
            else:    
                data_dict = {
                    "user_profile": sample['user_profile'],
                    "knowledge_graph": sample['knowledge_graph'],
                    "hop": sample['hop'],
                    "conversation": sample['conversation'],
                    "target": ACT + sample['target'][0] + TOPIC + sample['target'][1],
                    "plans": sample['plans']
                }
            data.append(data_dict)
    return data

def get_action_set(data_dir):
    action_path = os.path.join(data_dir, "label_action.txt")
    if os.path.isfile(action_path):
        action_set = []
        with open(action_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                act = line.strip()
                action_set.append(act)
        return action_set
    else:
        raise FileExistsError("{} not exist!".format(action_path))

def get_topic_set(data_dict):
    sub_obj = set()
    for s, p, o in data_dict['knowledge_graph']:
        sub_obj.add(s)
        sub_obj.add(o)
    return sub_obj

def fuzzy_str(with_unk, list_sen):
    string = with_unk.split("[UNK]")[0]
    for l in list_sen:
        if l[:len(string)] == string:
            return l
    return with_unk

def match_sentence(sen, list_sen):
    return_str = ACT
    now_is = True
    sen_split = []
    for s in sen.split(ACT)[1:]:
        sen_split.extend(s.split(TOPIC))
    for s in sen_split:
        if "[UNK]" in s:
            return_str = return_str + fuzzy_str(s, list_sen)
        else:
            return_str = return_str + s
        if now_is:
            return_str = return_str + TOPIC
            now_is = False
        else:
            return_str = return_str + ACT
            now_is = True
    return return_str[:-3]

def combine_tokens(output, tokenizer, vocab_list=None):
    return_sentence=[]
    for batch in range(output.size(0)):
        out_tokens = tokenizer.decode(output[batch, :]).split()
        return_tokens = []
        for t in out_tokens:
            if t == BOS:
                continue
            elif t == EOS or t == "[PAD]":
                break
            else:
                return_tokens.append(t)
        return_sentence.append("".join(return_tokens))
    if vocab_list is not None and "[UNK]" in return_sentence[0]:
        return_sentence = [match_sentence(return_sentence[0], vocab_list)]
    return return_sentence