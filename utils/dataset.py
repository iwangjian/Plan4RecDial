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
from utils.data_utils import CLS, SEP, ACT, TPC, BOP, EOP, BOS, EOS


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
    
    target_ids: List[int]
    input_ids: List[int]
    gold_ids: List[int]

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class DuRecDialDataset(Dataset):
    """
    Self-defined Dataset class for the DuRecDial dataset.
    Args:
        Dataset ([type]): [description]
    """
    def __init__(self,
        data_path, 
        tokenizer, 
        data_partition, 
        cache_dir=None, 
        max_seq_len=512, 
        max_plan_len=512, 
        turn_type_size=16,
        use_knowledge_hop=False,
        is_test=False
    ):
        self.tokenizer = tokenizer
        self.data_partition = data_partition
        
        self.cache_dir = cache_dir
        self.max_seq_len = max_seq_len
        self.max_plan_len = max_plan_len
        self.turn_type_size = turn_type_size
        self.use_knowledge_hop = use_knowledge_hop
        self.is_test = is_test
        
        self.instances = []
        self._cache_instances(data_path)

    def _cache_instances(self, data_path):
        """
        Load data tensors into memory or create the dataset when it does not exist.
        """
        signature = "{}_cache.pkl".format(self.data_partition)
        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, signature)
        else:
            cache_dir = os.mkdir("caches")
            cache_path = os.path.join(cache_dir, signature)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                logging.info("Loading cached instances from {}".format(cache_path))
                self.instances = pickle.load(f)
        else:          
            logging.info("Loading raw data from {}".format(data_path))
            all_samples = []
            with open(data_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    sample = json.loads(line.strip())
                    if self.is_test:
                        data_dict = {
                            "user_profile": sample["user_profile"],
                            "knowledge": sample["knowledge"],
                            "conversation": sample["conversation"],
                            "target": sample["target"],
                            "plan_path": None
                    }
                    else:
                        # construct reverse plan path
                        path_str = self._get_reverse_path(
                            sample["action_path"], sample["topic_path"],
                            target_action=sample["target"][0],
                            target_topic=sample["target"][1]
                        )
                        #print("target: ", sample["target"])
                        #print("action_path: ", sample["action_path"])
                        #print("topic_path: ", sample["topic_path"])
                        #print("path_str: {}\n".format(path_str))
                        data_dict = {
                            "user_profile": sample["user_profile"],
                            "knowledge": sample["knowledge"],
                            "conversation": sample["conversation"],
                            "target": sample["target"],
                            "plan_path": path_str
                        }
                    all_samples.append(data_dict)

            logging.info("Creating cache instances {}".format(signature))
            for row in tqdm(all_samples):
                p_ids, p_segs, p_poss = self._parse_profile(row["user_profile"])
                if self.use_knowledge_hop:
                    k_ids, k_segs, k_poss, k_hops = self._parse_hop_knowledge(row["knowledge"], row["target"])
                else:
                    k_ids, k_segs, k_poss, k_hops = self._parse_raw_knowledge(row["knowledge"])
                h_ids, h_segs, h_poss = self._parse_conversation(row["conversation"])
                target_ids, input_ids, gold_ids = self._parse_plan(row["plan_path"], row["target"])
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
                    "target_ids": target_ids,
                    "input_ids": input_ids,
                    "gold_ids": gold_ids
                }
                feature = InputFeature(**inputs)
                self.instances.append(feature)            
            with open(cache_path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))
    
    @staticmethod
    def _get_reverse_path(action_path: list, topic_path: list, target_action: str, target_topic: str):
        ptr = -1
        for idx in range(len(action_path)):
            if action_path[idx] == target_action and topic_path[idx] == target_topic:
                ptr = idx
                break
        if ptr > 0:
            action_path = action_path[:ptr+1]
            topic_path = topic_path[:ptr+1]
            action_path.reverse()
            topic_path.reverse()
        elif ptr == 0:
            action_path = [action_path[0]]
            topic_path = [topic_path[0]]
        else:
            action_path = [target_action] + action_path
            topic_path = [target_topic] + topic_path
        path_str = ""
        for a, t in zip(action_path, topic_path):
            if not t in path_str:
                path_str += "%s%s%s%s" % (ACT, a, TPC, t)
            elif not a in path_str:
                path_str += "%s%s%s%s" % (ACT, a, TPC, t)
        return path_str
    
    @staticmethod
    def _parse_kg_hop(kg_list, target):
        subjects = []
        for kg in kg_list:
            s, p, o = kg[0], kg[1], kg[2]
            if s not in subjects:
                subjects.append(s)
            if o not in subjects:
                subjects.append(o)
        num_subjects = len(subjects)
        target_topic = target[1]
        assert target_topic in subjects
        
        knowledge_hop = {}
        knowledge_hop[target_topic] = 1
        subjects.remove(target_topic)
        
        while len(subjects) != 0:
            break_flag = True
            for kg in kg_list:
                s, p, o = kg
                if s in knowledge_hop.keys():
                    if o not in knowledge_hop.keys():
                        knowledge_hop[o] = knowledge_hop[s] + 1
                        subjects.remove(o)
                        break_flag = False
                    else:
                        if knowledge_hop[o] > knowledge_hop[s] + 1:
                            knowledge_hop[o] = knowledge_hop[s] + 1
                            break_flag = False
                elif o in knowledge_hop.keys():
                    knowledge_hop[s] = knowledge_hop[o]+1
                    subjects.remove(s)
                    break_flag = False

            if break_flag:
                backup_topic = subjects[0]
                knowledge_hop[backup_topic] = 2
                subjects.remove(backup_topic)
                #print("backup_topic:", backup_topic)
                #print(subjects)
        assert len(knowledge_hop) == num_subjects
        return knowledge_hop

    def _parse_profile(self, profile):
        tokens, segs = [], []
        for k, v in profile.items():
            k_toks = self.tokenizer.tokenize(k)
            v_toks = self.tokenizer.tokenize(v)
            tokens = tokens + k_toks + v_toks + [SEP]
            segs = segs + len(k_toks)*[0] + len(v_toks)*[1] + [1]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(ids) > self.max_seq_len - 2:
            ids = ids[:self.max_seq_len - 2]
            segs = segs[:self.max_seq_len - 2]
            ids = self.tokenizer.convert_tokens_to_ids([CLS]) + ids + self.tokenizer.convert_tokens_to_ids([SEP])
            segs = [1] + segs + [1]
        else:
            ids = self.tokenizer.convert_tokens_to_ids([CLS]) + ids
            segs = [1] + segs
        poss = list(range(len(ids)))
        assert len(ids) == len(poss) and len(ids) == len(segs)
        return ids, segs, poss

    def _parse_hop_knowledge(self, knowledge, target):
        """parse knowledge with knowledge hops"""
        knowledge_hop = self._parse_kg_hop(knowledge, target)
        
        topics_dict = {}
        for s, p, o in knowledge:
            if s in topics_dict:
                topics_dict[s].append([p, o])
            else:
                topics_dict[s] = [[p, o]]
        max_hop = 0
        for k, v in knowledge_hop.items():
            if int(v) > max_hop:
                max_hop = int(v)
        
        tokens, segs, hops = [BOP], [0], [max_hop + 1]
        for k, v in topics_dict.items():
            k_ = self.tokenizer.tokenize(k)
            tokens = tokens + [TPC] + k_
            segs = segs + [0] + len(k_) * [0]
            hops = hops + (len(k_)+1) * [int(knowledge_hop.get(k, 0))]
            for p, o in v:
                p_ = self.tokenizer.tokenize(p)
                o_ = self.tokenizer.tokenize(o)
                tokens = tokens + p_ + o_
                segs = segs + len(p_)*[1] + len(o_)*[0]
                hops = hops + (len(p_)+len(o_))*[int(knowledge_hop.get(o, 0))]
        
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
    
    def _parse_raw_knowledge(self, knowledge):
        """parse knowledge by simple concat"""
        tokens, segs = [], []
        for kg in knowledge:
            kg_str = "".join(kg)
            kg_tok = self.tokenizer.tokenize(kg_str)
            tokens = tokens + kg_tok + [SEP]
            segs = segs + len(kg_tok) * [0] + [1]
        
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(ids) > self.max_seq_len - 1:
            ids = ids[:self.max_seq_len - 1]
            segs = segs[:self.max_seq_len - 1]
        hops = [0] * len(ids)
        poss = list(range(len(ids)))
        assert len(ids) == len(poss) == len(segs) == len(hops)
        return ids, segs, poss, hops
    
    def _parse_conversation(self, history):
        if len(history) > self.turn_type_size - 1:
            history = history[len(history) - (self.turn_type_size - 1):]
        cur_turn_type = len(history) % 2
        tokens, segs = [], []
        for h in history:
            h = self.tokenizer.tokenize(h)
            tokens = tokens + h + [SEP]
            segs = segs + len(h)*[cur_turn_type] + [cur_turn_type]
            cur_turn_type = cur_turn_type ^ 1
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(ids) > self.max_seq_len - 2:
            ids = ids[2 - self.max_seq_len:]
            segs = segs[2 - self.max_seq_len:]
            ids = self.tokenizer.convert_tokens_to_ids([CLS]) + ids + self.tokenizer.convert_tokens_to_ids([SEP])
            segs = [1] + segs + [1]
        else:
            ids = self.tokenizer.convert_tokens_to_ids([CLS]) + ids
            segs = [1] + segs
        poss = list(range(len(ids)))
        assert len(ids) == len(poss) == len(segs)
        return ids, segs, poss
    
    def _parse_plan(self, plan_path, target):
        target_str = ACT + target[0] + TPC + target[1]
        target_tokens = self.tokenizer.tokenize(target_str)
        
        if self.is_test:
            input_tokens = [BOS, ACT]
            gold_tokens = [BOS, ACT]
        else:
            plan_tokens = self.tokenizer.tokenize(plan_path)
            input_tokens = [BOS] + plan_tokens
            gold_tokens = plan_tokens + [EOS]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        gold_ids = self.tokenizer.convert_tokens_to_ids(gold_tokens)
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        assert len(input_ids) == len(gold_ids)

        return target_ids, input_ids, gold_ids

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]