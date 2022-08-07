# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100

class BartDataset(Dataset):
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
        # truncate previous tokens if dialogue context is too long
        if len(history) > self.max_seq_len - 1:
            input_ids = [self.bos] + history[-self.max_seq_len+1:]
        else:
            input_ids = [self.bos] + history
        decoder_input_ids = [self.bos] + response
        target_ids = response + [self.eos]
        
        instance = {}
        instance["input_ids"] = input_ids
        instance["decoder_input_ids"] = decoder_input_ids
        instance["labels"] = target_ids
        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        decoder_input_ids = pad_sequence(
            [torch.tensor(instance["decoder_input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=IGNORE_INDEX)
        
        return input_ids, decoder_input_ids, labels
