# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

USER = "[USER]"  # additional special token
BOT = "[BOT]"    # additional special token

NEW_ADD_TOKENS = ["[USER]", "[BOT]"]   # additional special tokens
IGNORE_INDEX = -100


class GPT2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=512, batch_first=True, lm_labels=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.bos = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.eos = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.lm_labels:
            history = self.data[index][0]
            response = self.data[index][1]
        else:
            history = self.data[index][0]
            response = []
        return self._process(history, response)

    def _process(self, history, response):
        # truncate previous tokens if dialogue history is too long
        if len(history) > self.max_seq_len - 1:
            history = history[-self.max_seq_len+1:]
        
        if self.lm_labels:
            input_ids = [self.bos] + history + response + [self.eos]
            lm_labels = [IGNORE_INDEX] * (len(history) + 1) + response + [self.eos]
        else:
            input_ids = [self.bos] + history
            lm_labels = [IGNORE_INDEX] * (len(history) + 1)

        instance = {}
        instance["input_ids"] = input_ids
        instance["lm_labels"] = lm_labels

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=IGNORE_INDEX)
        
        return input_ids, labels
