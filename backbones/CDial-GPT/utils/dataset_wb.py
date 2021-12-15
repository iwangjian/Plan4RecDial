# -*- coding: utf-8 -*-
# Some functions come from the Internet, if you violate your rights, please contact us.
from itertools import chain
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SPECIAL_TOKENS = ["[CLS]", "[PAD]", "[SEP]", "[speaker1]", "[speaker2]"]
IGNORE_INDEX = -100


class WBDataset(Dataset):

    def __init__(self, data, tokenizer, max_history=15, batch_first=True, lm_labels=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.lm_labels:
            history = self.data[index][-2 * self.max_history:-1]
            resposne = self.data[index][-1]
        else:
            history = self.data[index][-2 * self.max_history:-1]
            resposne = []

        return self.process(history, resposne)

    def process(self, history, resposne, with_eos=True, max_seq_length=512):
        bos, _, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        raw_sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
        
        # truncate previous tokens if dialogue history is too long
        if len(raw_sequence[1]) + len(raw_sequence[-1]) > max_seq_length - 1:
            max_history_len = max_seq_length - 1 - len(raw_sequence[-1])
            sequence = [raw_sequence[0]] + [raw_sequence[1][-max_history_len:]] + [raw_sequence[-1]]
        else:
            sequence = raw_sequence
        
        # handle token type ids for speaker1 and speaker2
        token_type_ids = [speaker1] * len(sequence[1])
        is_speaker2 = False
        for i, s in enumerate(sequence[1]):
            if s == speaker2:
                is_speaker2 = True
            elif s == speaker1:
                is_speaker2 = False
            if is_speaker2:
                token_type_ids[i] = speaker2

        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + token_type_ids + [speaker2] * len(sequence[-1])
        instance["lm_labels"] = [IGNORE_INDEX] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([IGNORE_INDEX] * sum(len(s) for s in sequence[:-1])) + sequence[-1]
        
        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=IGNORE_INDEX)
        return input_ids, token_type_ids, labels
