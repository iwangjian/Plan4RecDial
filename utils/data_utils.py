# -*- coding: utf-8 -*-
import os
import json
from transformers import BertTokenizer

PAD = "[PAD]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"

ACT = "[A]"       # denote an action
TPC = "[T]"       # denote a topic
BOP = "[BOP]"     # begin of knowledge hop
EOP = "[EOP]"     # end of knowledge hop
BOS = "[BOS]"     # begin of sequence
EOS = "[EOS]"     # end of sequence

SPECIAL_TOKENS_MAP = {"additional_special_tokens": [ACT, TPC, BOP, EOP, BOS, EOS]}

def get_tokenizer(config_dir):
    tokenizer = BertTokenizer.from_pretrained(config_dir)
    num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS_MAP)
    special_token_id_dict = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.convert_tokens_to_ids(BOS),
        "eos_token_id": tokenizer.convert_tokens_to_ids(EOS),
    }
    return tokenizer, num_added_tokens, special_token_id_dict

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

def get_topic_set(data_path):
    topic_set = []
    with open(data_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            sample = json.loads(line.strip())
            topic = set()
            for s, p, o in sample["knowledge"]:
                topic.add(s)
                topic.add(o)
            topic_set.append(topic)
    return topic_set

def fuzzy_str(with_unk, list_sen):
    string = with_unk.split(UNK)[0]
    for l in list_sen:
        if l[:len(string)] == string:
            return l
    return with_unk

def match_sentence(sen, list_sen):
    return_str = ACT
    now_is = True
    sen_split = []
    for s in sen.split(ACT)[1:]:
        sen_split.extend(s.split(TPC))
    for s in sen_split:
        if UNK in s:
            return_str = return_str + fuzzy_str(s, list_sen)
        else:
            return_str = return_str + s
        if now_is:
            return_str = return_str + TPC
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
            elif t == EOS or t == PAD:
                break
            elif t.upper() == "NULL":
                return_tokens.append("NULL")
            else:
                return_tokens.append(t)
        return_sentence.append("".join(return_tokens))
    if vocab_list is not None and UNK in return_sentence[0]:
        return_sentence = [match_sentence(return_sentence[0], vocab_list)]
    return return_sentence

def get_eval_output(path_str):
    # parse backward path
    # i.e., [A]...[T]...[A]act[T]tpc
    try:
        eval_span = path_str.split(ACT)[-1].strip()
    except IndexError:
        eval_span = None
    
    if eval_span is None:
        action, topic = UNK, UNK
    else:
        try:
            action = eval_span.split(TPC)[0].strip()
        except IndexError:
            action = UNK
        try:
            topic = eval_span.split(TPC)[-1].strip()
        except IndexError:
            topic = UNK
    return (action, topic)