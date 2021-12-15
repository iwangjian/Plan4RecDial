# -*- coding: utf-8 -*-
import torch


def lists_to_tensor3(list_l, max_len, pad_token_id):
    batch_len = len(list_l)
    return_ids = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    return_segs = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    return_poss = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    return_mask = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    for idx, l in enumerate(list_l):
        len_l = min([max_len, len(l[0])])
        return_ids[:len_l, idx] = torch.LongTensor(l[0][:len_l])
        return_segs[:len_l, idx] = torch.LongTensor(l[1][:len_l])
        return_poss[:len_l, idx] = torch.LongTensor(l[2][:len_l])
        return_mask[:len_l, idx] = torch.LongTensor([1]*len_l)
    return [return_ids, return_segs, return_poss, return_mask]

def lists_to_tensor3_plans(list_l, max_len, pad_token_id):
    batch_len = len(list_l)
    return_ids = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    return_segs = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    return_poss = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    return_mask = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    return_gold = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    for idx, l in enumerate(list_l):
        len_l = min([max_len, len(l[0])])
        return_ids[:len_l, idx] = torch.LongTensor(l[0][:len_l])
        return_segs[:len_l, idx] = torch.LongTensor(l[1][:len_l])
        return_poss[:len_l, idx] = torch.LongTensor(l[2][:len_l])
        return_mask[:len_l, idx] = torch.LongTensor([1]*len_l)
        return_gold[:len_l, idx] = torch.LongTensor(l[3][:len_l])
    return [return_ids, return_segs, return_poss, return_mask, return_gold]

def lists_to_tensor4(list_l, max_len, pad_token_id):
    batch_len = len(list_l)
    return_ids = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    return_segs = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    return_poss = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    return_hops = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    return_mask = torch.full((max_len, batch_len), pad_token_id, dtype=torch.int64)
    for idx, l in enumerate(list_l):
        len_l = min([max_len, len(l[0])])
        return_ids[:len_l, idx] = torch.LongTensor(l[0][:len_l])
        return_segs[:len_l, idx] = torch.LongTensor(l[1][:len_l])
        return_poss[:len_l, idx] = torch.LongTensor(l[2][:len_l])
        return_hops[:len_l, idx] = torch.LongTensor(l[3][:len_l])
        return_mask[:len_l, idx] = torch.LongTensor([1]*len_l)
    return [return_ids, return_segs, return_poss, return_hops, return_mask]

def max_length(list_l):
    return max(len(l) for l in list_l)


def custom_collate(mini_batch):
    """Custom collate function for dealing with batches of input data.
    Arguments:
        mini_batch: A list of input features.
    Return:
        dict: (dict) A dict of tensor.
    """
    pad_token_id = 0   # Note: pad_token_id should be in consistent with tokenizer
    
    up_ids, kg_ids, hs_ids, pl_ids, tg_ids = [], [], [], [], []
    for sample in mini_batch:
        up_ids.append(sample.user_profile_ids)
        kg_ids.append(sample.knowledge_ids)
        hs_ids.append(sample.conversation_ids)
        pl_ids.append(sample.plan_ids)
        tg_ids.append(sample.target_ids)
    max_up_len, max_kg_len, max_hs_len, max_pl_len, max_tg_len = max_length(up_ids), max_length(kg_ids), max_length(hs_ids), max_length(pl_ids), max_length(tg_ids)
    up_list, kg_list, hs_list, pl_list, tg_list = [], [], [], [], []
    for sample in mini_batch:
        up_list.append([sample.user_profile_ids, sample.user_profile_segs, sample.user_profile_poss])
        kg_list.append([sample.knowledge_ids, sample.knowledge_segs, sample.knowledge_poss, sample.knowledge_hops])
        hs_list.append([sample.conversation_ids, sample.conversation_segs, sample.conversation_poss])
        pl_list.append([sample.plan_ids, sample.plan_segs, sample.plan_poss, sample.ground_truth])
        tg_list.append([sample.target_ids, sample.target_segs, sample.target_poss])
    collated_batch = {}
    collated_batch['user_profile'] = lists_to_tensor3(up_list, max_up_len, pad_token_id)
    collated_batch['knowledge'] = lists_to_tensor4(kg_list, max_kg_len, pad_token_id)
    collated_batch['conversation'] = lists_to_tensor3(hs_list, max_hs_len, pad_token_id)
    collated_batch['plans'] = lists_to_tensor3_plans(pl_list, max_pl_len, pad_token_id)
    collated_batch['target'] = lists_to_tensor3(tg_list, max_tg_len, pad_token_id)
    
    return collated_batch
