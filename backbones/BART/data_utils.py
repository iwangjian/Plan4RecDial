# -*- coding: utf-8 -*-
import os
import json
import pickle

SEP = "[SEP]"
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
    if center_topic == "NULL":
        for triple in kg_list:
            s, p, o = triple
            if "气温:" in s or "气温:" in o:
                sub_kg.append(triple)
    else:
        for triple in kg_list:
            s, p, o = triple
            if p == "新闻":
                sub_kg.append(triple)
            elif s.lower() == center_topic.lower() or o.lower() == center_topic.lower():
                if not triple in sub_kg:
                    sub_kg.append(triple)
    return sub_kg

def convert_data(fp, extract_kg=False, tcp_path=None):
    cur_actions, cur_topics = [], []
    if tcp_path is not None:
        with open(tcp_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                sample = json.loads(line)
                action = sample["cur_action"]
                topic = sample["cur_topic"]
                topic = topic.replace("null", "NULL")
                cur_actions.append(action)
                cur_topics.append(topic)
    data = []
    with open(fp, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            sample = json.loads(line)
            original_goal = sample["original_goal"]
            user_profile = sample["user_profile"]
            history = sample["conversation"]
            resp_str = sample["response"]
            
            input_str = ""
            if extract_kg:
                if tcp_path is not None:
                    # extract knowledge according to generated plans
                    kg_list = extract_knowledge(sample["knowledge_graph"], cur_topics[idx])
                    for triple in kg_list:
                        kd = "".join(triple)
                        input_str += kd
                        input_str += SEP
                    input_str += cur_actions[idx] + cur_topics[idx] + SEP
                else:
                    # extract knowledge according to current labeled topic
                    kg_list = extract_knowledge(sample["knowledge_graph"], sample["cur_topic"])
                    for triple in kg_list:
                        kd = "".join(triple)
                        input_str += kd
                        input_str += SEP
                    input_str += sample["cur_action"] + sample["cur_topic"] + SEP
            else:
                kg_list = sample["knowledge_graph"]
                for triple in kg_list:
                    kd = "".join(triple)
                    input_str += kd
                    input_str += SEP
                input_str += sample["target"][0] + sample["target"][1] + SEP
            
            for k, v in user_profile.items():
                input_str += k
                input_str += v
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


def load_data(tokenizer, logger, dataset_path, cache_dir, data_partition="train", use_tcp=False, tcp_path=None):
    """ Load data from cache or create from raw data."""
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
            if data_partition == "test":
                assert tcp_path is not None
                logger.info("Loading from [{}] to prepare test data for TCP-enhanced generation.".format(tcp_path))
                # prepare test data for TCP-enhanced Bart generation
                data = convert_data(fp=dataset_path, extract_kg=True, tcp_path=tcp_path)
            else:
                logger.info("Prepare train/valid data for TCP-enhanced generation.")
                # prepare train/valid data for TCP-enhanced Bart fine-tuning
                data = convert_data(fp=dataset_path, extract_kg=True)
        else:
            # prepare data for Bart fine-tuning
            data = convert_data(fp=dataset_path)  
        
        # tokenize data
        logger.info("Tokenizing ...")
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
