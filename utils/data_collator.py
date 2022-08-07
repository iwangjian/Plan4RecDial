# -*- coding: utf-8 -*-
import torch


def max_seq_length(list_l):
    return max(len(l) for l in list_l)

def pad_sequence(list_l, max_len, padding_value=0):
    assert len(list_l) <= max_len
    padding_l = [padding_value] * (max_len - len(list_l))
    padded_list = list_l + padding_l
    return padded_list


class PlanCollator(object):
    """
    Data collator for planning
    """
    def __init__(self, device, padding_idx=0):
        self.device = device
        self.padding_idx = padding_idx
    
    def list_to_tensor(self, list_l):
        max_len = max_seq_length(list_l)
        padded_lists = []
        for list_seq in list_l:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=self.padding_idx))
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor

    def varlist_to_tensor(self, list_vl):
        lens = []
        for list_l in list_vl:
            lens.append(max_seq_length(list_l))
        max_len = max(lens)
        
        padded_lists = []
        for list_seqs in list_vl:
            v_list = []
            for list_l in list_seqs:
                v_list.append(pad_sequence(list_l, max_len, padding_value=self.padding_idx))
            padded_lists.append(v_list)
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor
    
    def get_attention_mask(self, data_tensor: torch.tensor):
        attention_mask = data_tensor.masked_fill(data_tensor == self.padding_idx, 0)
        attention_mask = attention_mask.masked_fill(attention_mask != self.padding_idx, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask
    
    def custom_collate(self, mini_batch):
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        up_ids = []
        kg_ids, kg_segs, kg_poss, kg_hops = [], [], [], []
        hs_ids, hs_segs, hs_poss = [], [], []
        tg_ids = []
        input_ids, gold_ids = [], []
        for sample in mini_batch:
            up_ids.append(sample.user_profile_ids)
            kg_ids.append(sample.knowledge_ids)
            kg_segs.append(sample.knowledge_segs)
            kg_poss.append(sample.knowledge_poss)
            kg_hops.append(sample.knowledge_hops)
            hs_ids.append(sample.conversation_ids)
            hs_segs.append(sample.conversation_segs)
            hs_poss.append(sample.conversation_poss)

            tg_ids.append(sample.target_ids)
            input_ids.append(sample.input_ids)
            gold_ids.append(sample.gold_ids)
            
        
        batch_up_ids = self.list_to_tensor(up_ids)
        batch_up_masks = self.get_attention_mask(batch_up_ids)
        
        batch_kg_ids = self.list_to_tensor(kg_ids)
        batch_kg_segs = self.list_to_tensor(kg_segs)
        batch_kg_poss = self.list_to_tensor(kg_poss)
        batch_kg_hops = self.list_to_tensor(kg_hops)
        batch_kg_masks = self.get_attention_mask(batch_kg_ids)

        batch_hs_ids = self.list_to_tensor(hs_ids)
        batch_hs_segs = self.list_to_tensor(hs_segs)
        batch_hs_poss = self.list_to_tensor(hs_poss)
        batch_hs_masks = self.get_attention_mask(batch_hs_ids)

        batch_tg_ids = self.list_to_tensor(tg_ids)
        batch_tg_masks = self.get_attention_mask(batch_tg_ids)

        batch_input_ids = self.list_to_tensor(input_ids)
        batch_input_masks = self.get_attention_mask(batch_input_ids)
        batch_gold_ids = self.list_to_tensor(gold_ids)

        collated_batch = {
            "user_profile": [batch_up_ids, batch_up_masks],
            "knowledge": [batch_kg_ids, batch_kg_segs, batch_kg_poss, batch_kg_hops, batch_kg_masks],
            "conversation": [batch_hs_ids, batch_hs_segs, batch_hs_poss, batch_hs_masks],
            "target": [batch_tg_ids, batch_tg_masks],
            "plan": [batch_input_ids, batch_input_masks, batch_gold_ids]
        }

        return collated_batch
