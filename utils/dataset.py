# -*- coding: utf-8 -*-
import logging
import os
import pickle
import dataclasses
import json
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import ACT, TOPIC, BOP, EOP, BOS, EOS


@dataclass(frozen=True)
class InputFeature:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    user_profile_ids: List[int]
    user_profile_segs: List[int]
    user_profile_poss: List[int]
    knowledge_ids: List[int]
    knowledge_segs: List[int]
    knowledge_poss: List[int]
    knowledge_hops: List[int]
    conversation_ids: List[int]
    conversation_segs: List[int]
    conversation_poss: List[int]
    plan_ids: List[int]
    plan_segs: List[int]
    plan_poss: List[int]
    target_ids: List[int]
    target_segs: List[int]
    target_poss: List[int]
    ground_truth: List[int]

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class DuRecDialDataset(Dataset):
    """
    Self-defined Dataset class for the DuRecDial dataset.
    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, data, tokenizer, data_partition, cache_dir=None, max_seq_len=512, max_plan_len=512, turn_type_size=16):
        self.data = data
        self.tokenizer = tokenizer
        self.data_partition = data_partition
        assert self.data_partition in ("train", "dev", "test")
        
        self.cache_dir = cache_dir
        self.max_seq_len = max_seq_len
        self.max_plan_len = max_plan_len
        self.turn_type_size = turn_type_size
        
        self.instances = []
        self._cache_instances()

    def _cache_instances(self):
        """
        Load data tensors into memory or create the dataset when it does not exist.
        """
        signature = "{}_cache.pkl".format(self.data_partition)
        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, signature)
        else:
            cache_dir = os.mkdir(".cache")
            cache_path = os.path.join(cache_dir, signature)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                logging.info("Loading cached instances from {}".format(cache_path))
                self.instances = pickle.load(f)
        else:          
            logging.info("Creating cache instances {}".format(signature))
            for row in tqdm(self.data):
                p_ids, p_segs, p_poss = self._user_profile_parse(row['user_profile'])
                k_ids, k_segs, k_poss, k_hops = self._knowledge_parse(knowledge=row['knowledge_graph'], hop=row['hop'])
                h_ids, h_segs, h_poss = self._conversation_parse(history=row['conversation'])
                plan_ids, plan_segs, plan_poss, target_ids, target_segs, target_poss, gold_ids = self._plan_parse(target=row['target'], plans=row['plans'])
                inputs = {
                    "user_profile_ids": p_ids,
                    "user_profile_segs": p_segs,
                    "user_profile_poss": p_poss,
                    "knowledge_ids": k_ids,
                    "knowledge_segs": k_segs,
                    "knowledge_poss": k_poss,
                    "knowledge_hops": k_hops,
                    "conversation_ids": h_ids,
                    "conversation_segs": h_segs,
                    "conversation_poss": h_poss,
                    "plan_ids": plan_ids,
                    "plan_segs": plan_segs,
                    "plan_poss": plan_poss,
                    "target_ids": target_ids,
                    "target_segs": target_segs,
                    "target_poss": target_poss,
                    "ground_truth": gold_ids
                }
                feature = InputFeature(**inputs)
                self.instances.append(feature)            
            with open(cache_path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))
    
    def _user_profile_parse(self, profile):
        tokens, segs = [], []
        for key, value in profile.items():
            key = self.tokenizer.tokenize(key.replace(" ", "").lower())
            value = self.tokenizer.tokenize(value.replace(" ", "").lower())
            tokens = tokens + key + value + ['[SEP]']
            segs = segs + len(key)*[0] + len(value)*[1] + [1]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(ids) > self.max_seq_len - 2:
            ids = ids[:self.max_seq_len - 2]
            segs = segs[:self.max_seq_len - 2]
            ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"]) + ids + self.tokenizer.convert_tokens_to_ids(["[SEP]"])
            segs = [1] + segs + [1]
        else:
            ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"]) + ids
            segs = [1] + segs
        poss = list(range(len(ids)))
        assert len(ids) == len(poss) and len(ids) == len(segs)
        return ids, segs, poss
    
    def _knowledge_parse(self, knowledge, hop):
        topics_dict = {}
        for s, p, o in knowledge:
            if s in topics_dict:
                topics_dict[s].append([p, o])
            else:
                topics_dict[s] = [[p, o]]
        max_hop = 0
        for key, value in hop.items():
            if int(value) > max_hop:
                max_hop = int(value)
        
        tokens, segs, hops = [BOP], [0], [max_hop + 1]
        for key, value in topics_dict.items():
            key_ = self.tokenizer.tokenize(key.replace(" ", "").lower())
            tokens = tokens + [TOPIC] + key_
            segs = segs + [0] + len(key_) * [0]
            hops = hops + (len(key_)+1) * [int(hop[key])]
            for p, o in value:
                p_ = self.tokenizer.tokenize(p.replace(" ", "").lower())
                o_ = self.tokenizer.tokenize(o.replace(" ", "").lower())
                tokens = tokens + p_ + o_
                segs = segs + len(p_)*[1] + len(o_)*[0]
                hops = hops + (len(p_)+len(o_))*[int(hop[o])]
        
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(ids) > self.max_seq_len - 1:
            ids = ids[:self.max_seq_len - 1]
            segs = segs[:self.max_seq_len - 1]
            hops = hops[:self.max_seq_len - 1]
        ids = ids + self.tokenizer.convert_tokens_to_ids([EOP])
        segs = segs + [0]
        hops = hops + [0]
        poss = list(range(len(ids)))
        assert len(ids) == len(poss) == len(segs) == len(hops)
        return ids, segs, poss, hops
    
    def _conversation_parse(self, history):
        def delete_id(h):
            if h[0]=='[' and h[2]==']':
                return h[3:]
            return h
        history = [delete_id(h) for h in history]
        if len(history) > self.turn_type_size - 1:
            history = history[len(history) - (self.turn_type_size - 1):]
        cur_turn_type = len(history) % 2
        tokens, segs = [], []
        for h in history:
            h = self.tokenizer.tokenize(h.replace(" ","").lower())
            tokens = tokens + h + ['[SEP]']
            segs = segs + len(h)*[cur_turn_type] + [cur_turn_type]
            cur_turn_type = cur_turn_type ^ 1
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(ids) > self.max_seq_len - 2:
            ids = ids[2 - self.max_seq_len:]
            segs = segs[2 - self.max_seq_len:]
            ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"]) + ids + self.tokenizer.convert_tokens_to_ids(["[SEP]"])
            segs = [1] + segs + [1]
        else:
            ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"]) + ids
            segs = [1] + segs
        poss = list(range(len(ids)))
        assert len(ids) == len(poss) == len(segs)
        return ids, segs, poss
    
    def _plan_parse(self, target, plans):
        target_tokens = self.tokenizer.tokenize(target.replace(" ", "").lower())
        plans_tokens = self.tokenizer.tokenize(plans.replace(" ", "").lower())
        if self.data_partition == "test":
            final_tokens = [BOS, ACT]
            groundtruth = [ACT, ACT]
        else:
            final_tokens = [BOS] + plans_tokens
            groundtruth = plans_tokens + [EOS]
        ids = self.tokenizer.convert_tokens_to_ids(final_tokens)
        gold_ids = self.tokenizer.convert_tokens_to_ids(groundtruth)
        segs = len(final_tokens)*[0]
        poss = list(range(len(ids)))
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        target_segs = len(target_tokens)*[0]
        target_poss = list(range(len(target_ids)))
        assert len(ids) == len(poss) == len(segs) == len(gold_ids)
        assert len(target_ids )== len(target_poss) == len(target_segs)
        return ids, segs, poss, target_ids, target_segs, target_poss, gold_ids

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]