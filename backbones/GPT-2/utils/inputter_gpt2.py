# -*- coding: utf-8 -*-
import os
import json
import torch
import pickle
from torch.utils.data import DataLoader
from utils.dataset_gpt2 import GPT2Dataset

SEP = ";"
USER = "[USER]"  # additional special token
BOT = "[BOT]"    # additional special token

def remove_goal(history):
    if "[1]" in history:
        history = history.replace("[1]", "")
    elif "[2]" in history:
        history = history.replace("[2]", "")
    elif "[3]" in history:
        history = history.replace("[3]", "")
    elif "[4]" in history:
        history = history.replace("[4]", "")
    elif "[5]" in history:
        history = history.replace("[5]", "")
    elif "[6]" in history:
        history = history.replace("[6]", "")
    elif "[7]" in history:
        history = history.replace("[7]", "")
    elif "[8]" in history:
        history = history.replace("[8]", "")
    return history

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

def convert_data(fp, extract_kg=False, tcp_path=None):
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
    with open(fp, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            sample = json.loads(line)
            original_goal = sample["original_goal"]
            user_profile = sample["user_profile"]
            history = sample["conversation"]
            resp_str = sample["response"]
            
            if extract_kg:
                if tcp_path is not None:
                    # extract knowledge according to generated plans
                    kg_list = extract_knowledge(sample["knowledge_graph"], cur_topics[idx])
                else:
                    # extract knowledge according to current labeled topic
                    kg_list = extract_knowledge(sample["knowledge_graph"], sample["cur_topic"])
            else:
                kg_list = sample["knowledge_graph"]
            
            input_str = ""
            for k, v in user_profile.items():
                input_str += k
                input_str += v
                input_str += SEP
            for triple in kg_list:
                kd = "".join(triple)
                input_str += kd
                input_str += SEP
            
            led_by_bot = False
            if "Bot主动" in original_goal[0]:
                led_by_bot = True
            for hdx, utt in enumerate(history):
                if hdx % 2 == 0:
                    if led_by_bot:
                        input_str += BOT
                    else:
                        input_str += USER
                else:
                    if led_by_bot:
                        input_str += USER
                    else:
                        input_str += BOT
                input_str += remove_goal(utt)
            input_str += BOT
            data.append([input_str, resp_str])
    return data


def tokenize(tokenizer, obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(tokenizer, o)) for n, o in obj.items())
    return list(tokenize(tokenizer, o) for o in obj)


def get_data(tokenizer, logger, dataset_path, cache_dir, data_partition="train", use_tcp=False, tcp_path=None):
    """ Get data from cache or create from raw data."""
    if use_tcp:
        cache_dir = cache_dir + '_w_TCP'
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, "{}_cache.pkl".format(data_partition))
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            logger.info("Loading cached data from [{}]".format(cache_path))
            tokenized_data = pickle.load(f)
    else:          
        logger.info("Creating cached data [{}]".format(cache_path))

        if use_tcp:
            if tcp_path is not None:
                # prepare test data for TCP-enhanced GPT-2 generation
                data = convert_data(fp=dataset_path, extract_kg=True, tcp_path=tcp_path)
            else:
                # prepare train/valid data for TCP-enhanced GPT-2 fine-tuning
                data = convert_data(fp=dataset_path, extract_kg=True)
        else:
            # prepare data for GPT-2 fine-tuning
            data = convert_data(fp=dataset_path)  
        
        # tokenize data
        tokenized_data = tokenize(tokenizer, data)

        # caching data
        with open(cache_path, 'wb') as f:
            pickle.dump(tokenized_data, f)
        
        ####################################################
        # for debugging
        data_dict = {data_partition: data}
        save_fp = os.path.join(cache_dir, "{}_cache.json".format(data_partition))
        with open(save_fp, 'w', encoding='utf-8') as fw:
            json.dump(data_dict, fw, ensure_ascii=False, indent=0)
        ####################################################
    logger.info("Total of {} instances were cached.".format(len(tokenized_data)))
    return tokenized_data


def build_dataloaders(args, tokenizer, logger):
    logger.info("Build train and validation dataloaders")

    train_data = get_data(tokenizer, logger, args.train_path, args.cache_dir, 
        data_partition="train", use_tcp=args.use_tcp)
    valid_data = get_data(tokenizer, logger, args.valid_path, args.cache_dir, 
        data_partition="valid", use_tcp=args.use_tcp)
    
    train_dataset = GPT2Dataset(train_data, tokenizer, max_history=args.max_history)
    valid_dataset = GPT2Dataset(valid_data, tokenizer, max_history=args.max_history)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              collate_fn=train_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler,
                              collate_fn=valid_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.valid_batch_size,
                              shuffle=False)
    
    return train_loader, valid_loader, train_sampler, valid_sampler
