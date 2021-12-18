#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import argparse

SEP = ";"
TURN_SEP = " __eou__ "  # Note: it has spaces in order to adapt ERNIE-GEN reader


def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def delete_id(h):
    if h[0]=='[' and h[2]==']':
        return h[3:]
    return h

def conv_parse(conv_list):
    conv_list = [delete_id(c) for c in conv_list]
    return TURN_SEP.join(conv_list)

def extract_knowledge(kg_list, center_topic):
    """Extract knowledge according to the center topic"""
    sub_kg = []
    if center_topic == "NULL" or center_topic == "null":
        return sub_kg
    
    for triple in kg_list:
        s, p, o = triple
        if s.lower() == center_topic.lower() or o.lower() == center_topic.lower():
            sub_kg.append(triple)
    return sub_kg

def convert_data(data_path, extract_kg=False, tcp_path=None):
    cur_topics = []
    if tcp_path is not None:
        with open(tcp_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                sample = json.loads(line)
                act_sep = "[a]"
                top_sep = "[t]"
                plans = sample["plans"].lower()
                try:
                    topic = plans.split(top_sep)[1].split(act_sep)[0].strip()
                except IndexError:
                    topic = "None"
                cur_topics.append(topic)
    data = []
    with open(data_path, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            sample = json.loads(line)
            user_profile = sample["user_profile"]
    
            if extract_kg:
                if tcp_path is not None:
                    # extract knowledge according to generated plans
                    kg_list = extract_knowledge(sample["knowledge_graph"], cur_topics[idx])
                else:
                    # extract knowledge according to current labeled topic
                    kg_list = extract_knowledge(sample["knowledge_graph"], sample["cur_topic"])
            else:
                kg_list = sample["knowledge_graph"]
            
            knowledge = ""
            for k, v in user_profile.items():
                knowledge += k
                knowledge += v
                knowledge += SEP
            for triple in kg_list:
                kd = "".join(triple)
                knowledge += kd
                knowledge += SEP
            
            src = conv_parse(sample["conversation"])
            tgt = sample["response"]
            dict_sample = {
                "knowledge": knowledge,
                "src": src,
                "tgt": tgt
            }
            data.append(dict_sample)
    return data

def save_data(data, data_partition, cache_dir):
    assert data_partition in ("train", "valid", "test")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    save_path = os.path.join(cache_dir, "{}.tsv".format(data_partition))

    with open(save_path, 'w', encoding='utf-8') as fw:
        fw.write("knowledge\tsrc\ttgt\n")
        for sample in data:
            fw.write(sample["knowledge"]+"\t"+sample["src"]+"\t"+sample["tgt"]+"\n")
    print("{}: Saved total {} samples to [{}]".format(data_partition.upper(), len(data), save_path))


def main(args):
    if args.use_tcp:
        print("Preprocessing data for TCP-enhanced ERNIE-GEN...")
        assert args.tcp_path is not None and os.path.isfile(args.tcp_path), "`tcp_path` is not a valid file path!"
        train_data = convert_data(args.train_path, extract_kg=True)
        valid_data = convert_data(args.valid_path, extract_kg=True)
        test_data = convert_data(args.test_path, extract_kg=True, tcp_path=args.tcp_path)
    else:
        print("Preprocessing data for raw ERNIE-GEN...")
        train_data = convert_data(args.train_path)
        valid_data = convert_data(args.valid_path)
        test_data = convert_data(args.test_path)
    
    cache_dir = args.cache_dir + '_w_TCP' if args.use_tcp else args.cache_dir
    
    save_data(train_data, data_partition="train", cache_dir=cache_dir)
    save_data(valid_data, data_partition="valid", cache_dir=cache_dir)
    save_data(test_data, data_partition="test", cache_dir=cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_tcp', type=str2bool, default="False", help="Whether or not use TCP-enhanced generation.")
    parser.add_argument('--tcp_path', type=str, default=None, help="Path of the decoded plans by TCP.")
    parser.add_argument("--train_path", type=str, default=None, help="Path of the train data.")
    parser.add_argument("--valid_path", type=str, default=None, help="Path of the valid data.")
    parser.add_argument("--test_path", type=str, default=None, help="Path of the test data.")
    parser.add_argument("--cache_dir", type=str, default="caches",help="Path of the dataset cache dir.")
    args = parser.parse_args()

    main(args=args)
