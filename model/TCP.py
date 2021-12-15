# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import copy
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Iterable, Callable, List
from model.GAT import GraphAttenTransformer
from model.MemNet import UserMemory
from model.Planner import Planner
from utils.data_utils import ACT, TOPIC

from transformers import (
    BertModel,
    BertTokenizer,
    LogitsProcessorList, 
    HammingDiversityLogitsProcessor, 
    NoBadWordsLogitsProcessor, 
    MinLengthLogitsProcessor, 
    PrefixConstrainedLogitsProcessor, 
    ForcedBOSTokenLogitsProcessor, 
    ForcedEOSTokenLogitsProcessor, 
    InfNanRemoveLogitsProcessor, 
    RepetitionPenaltyLogitsProcessor, 
    NoRepeatNGramLogitsProcessor, 
    StoppingCriteriaList, 
    MaxLengthCriteria, 
    MaxTimeCriteria,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class TCP(nn.Module):
    """
    Model class: Target-driven Conversation Planning (TCP)
    Args:
        args (Namespace): All necessary arguments for the model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.pad_token_id = args.pad_token_id
        self.bos_token_id = args.bos_token_id
        self.eos_token_id = args.eos_token_id
        self.forced_bos_token_id = args.forced_bos_token_id
        self.forced_eos_token_id = args.forced_eos_token_id

        self.kg_encoder = GraphAttenTransformer.from_pretrained(args.bert_dir)
        self.kg_encoder.resize_token_embeddings(self.vocab_size)

        self.conv_encoder = BertModel.from_pretrained(args.bert_dir)
        self.conv_encoder.resize_token_embeddings(self.vocab_size)

        self.user_memnet = UserMemory(vocab_size=self.vocab_size,
            query_size=args.embed_dim, memory_size=args.embed_dim, max_hop=args.max_memory_hop, 
            dropout=args.dropout, init_std=args.init_std,
            padding_idx=args.pad_token_id)
        
        self.planner = Planner(args=args)
        self.lm_vocab = nn.Linear(args.embed_dim, self.vocab_size, bias=True)
    
    def _init_all_weights(self):
        self.lm_vocab.weight.data.normal_(mean=0.0, std=self.args.init_std)
        if self.lm_vocab.bias is not None:
            self.lm_vocab.bias.data.zero_()

    def forward(self, batch, is_test=False):
        """model training"""
        up_ids, _, _, up_mask = batch['user_profile']
        kg_ids, kg_segs, kg_poss, kg_hops, kg_mask = batch['knowledge']
        hs_ids, hs_segs, hs_poss, hs_mask = batch['conversation']
        pl_ids, _, _, pl_mask, gold_ids = batch['plans']
        tg_ids, _, _, tg_mask = batch['target']
        
        kg_output = self.kg_encoder(
            input_ids=kg_ids, 
            attention_mask=kg_mask, 
            token_type_ids=kg_segs, 
            position_ids=kg_poss, 
            hops_ids=kg_hops)[0]
        conv_output = self.conv_encoder(
            input_ids=hs_ids, 
            attention_mask=hs_mask, 
            token_type_ids=hs_segs, 
            position_ids=hs_poss)[0]
        user_mem = self.user_memnet.load_memory(up_ids, conv_output)
        user_mem_output = self.user_memnet(
            context=conv_output, 
            user_memory_list=user_mem, 
            mask=hs_mask)
        
        planner_output = self.planner(
            input_ids=pl_ids,
            attention_mask=pl_mask,
            target_ids=tg_ids,
            target_mask=tg_mask,
            kg_encoder_hidden_states=kg_output,
            kg_encoder_attention_mask=kg_mask,
            up_encoder_hidden_states=user_mem_output,
            up_encoder_attention_mask=up_mask,
            hs_encoder_hidden_states=conv_output,
            hs_encoder_attention_mask=hs_mask,
        )
        lm_scores = self.lm_vocab(planner_output[0]).contiguous()
        
        if is_test:
            output = {
                "lm_scores": lm_scores,
            }
        else:
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(lm_scores.view(-1, self.vocab_size), gold_ids.view(-1))
            pred = torch.softmax(lm_scores, -1)
            _, pred_y = pred.max(-1)
            acc = (torch.eq(pred_y, gold_ids).float()*pl_mask).sum().item()
            tot_tokens = pl_mask.float().sum().item()
            output = {
                "lm_scores": lm_scores,
                "loss": lm_loss,
                "accuracy": acc,
                "total_tokens": tot_tokens
            }
        return output

    def generate(self, args, inputs, action_set=None, topic_set=None, tokenizer: Optional[BertTokenizer]=None):
        """model inference"""
        # parse args and set accordingly if need
        decoding_mode = args.decoding_mode
        assert decoding_mode in ("greedy", "topk", "topp")
        max_length = args.max_length
        assert max_length > 0

        use_ssd = args.use_ssd or False
        min_length = args.min_length or 0
        top_k = args.top_k or 1
        top_p = args.top_p or 1.0
        repetition_penalty = args.repetition_penalty or None
        diversity_penalty = args.diversity_penalty or None
        no_repeat_ngram_size = args.no_repeat_ngram_size or None
        bad_words_ids = args.bad_words_ids or None
        remove_invalid_values = args.remove_invalid_values or False

        # determine decoding mode
        if decoding_mode == "topk":
            is_topk_sampling = (top_k > 0)
        else:
            is_topk_sampling = False
        if decoding_mode == "topp":
            is_topp_sampling = (top_p > 0 and top_p <= 1.0)
        else:
            is_topp_sampling = False

        # get logits processor
        logits_processor = get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=self.eos_token_id,
            forced_bos_token_id=self.forced_bos_token_id,
            forced_eos_token_id=self.forced_eos_token_id,
            prefix_allowed_tokens_fn=None,
            num_beams=1,
            num_beam_groups=1,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
        )
        # get decoding stopping criteria
        stopping_criteria = get_stopping_criteria(max_length=max_length, max_time=None)
        
        if is_topk_sampling:
            return self.topk_sampling(
                inputs=inputs,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                max_length=max_length,
                top_k=top_k
            )
        elif is_topp_sampling:
            return self.topp_sampling(
                inputs=inputs,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                max_length=max_length,
                top_p=top_p
            )
        else:
            # use greedy search decoding as default
            return self.greedy_search(
                inputs=inputs,
                use_ssd=use_ssd,
                action_set=action_set,
                topic_set=topic_set,
                tokenizer=tokenizer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                max_length=max_length
            )

    def topk_sampling(self,
        inputs,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_scores: Optional[bool] = False,
        return_dict_in_generate: Optional[bool] = False,
        top_k: int = 0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
        ) -> torch.FloatTensor:
        
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.eos_token_id
        scores = () if (return_dict_in_generate and output_scores) else None
        
        input_ids = inputs['plans'][0]
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        while True:
            inputs["plans"][0], inputs["plans"][3] = input_ids, None
            # compute logits
            model_out = self(inputs, is_test=True)
            next_token_logits = model_out["lm_scores"][:, -1, :]
            if top_k > 0:
                next_token_logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
                    None, next_token_logits
                )
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            next_tokens_scores = F.softmax(next_tokens_scores, dim=-1)
            next_tokens = torch.multinomial(next_tokens_scores, 1)
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break
        return input_ids

    def topp_sampling(self,
        inputs,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_scores: Optional[bool] = False,
        return_dict_in_generate: Optional[bool] = False,
        top_p: float = 0.9,
        min_tokens_to_keep: int = 1,
        ) -> torch.FloatTensor:
        
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.eos_token_id
        scores = () if (return_dict_in_generate and output_scores) else None
        
        input_ids = inputs['plans'][0]
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        while True:
            inputs["plans"][0], inputs["plans"][3] = input_ids, None
            # compute logits
            model_out = self(inputs, is_test=True)
            next_token_logits = model_out["lm_scores"][:, -1, :]
            if 0 < top_p <= 1.0:
                next_token_logits = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)(
                    None, next_token_logits
                )
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            next_tokens_scores = F.softmax(next_tokens_scores, dim=-1)
            next_tokens = torch.multinomial(next_tokens_scores, 1)
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break
        return input_ids

    def greedy_search(self,
        inputs,
        use_ssd: bool = False,
        action_set: Optional[Iterable[str]] = None,
        topic_set: Optional[Iterable[str]] = None,
        tokenizer: Optional[BertTokenizer] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_scores: Optional[bool] = False,
        return_dict_in_generate: Optional[bool] = False,
    ) -> torch.LongTensor:
        
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.eos_token_id
        scores = () if (return_dict_in_generate and output_scores) else None
        
        input_ids = inputs['plans'][0]
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        
        def set_search(string, str_list):
            num = 0
            gold = None
            for l in str_list:
                if l[:len(string)].lower() == string.lower():
                    num += 1
                    gold = l
            if num == 1:
                return gold[len(string)-1:]
            else:
                return None
        
        if use_ssd:
            action_list = [a+TOPIC for a in action_set] if action_set is not None else []
            topic_list = [t+ACT for t in topic_set] if topic_set is not None else []
            span = ""
            now_type = "Action"
            total_str = ACT
            while True:
                inputs["plans"][0], inputs["plans"][3] = input_ids, None
                # compute logits
                model_out = self(inputs, is_test=True)
                next_token_logits = model_out["lm_scores"][:, -1, :]
                next_tokens_scores = logits_processor(input_ids, next_token_logits)
                next_tokens = torch.argmax(next_tokens_scores, dim=-1)
                if next_tokens[0] == eos_token_id or input_ids.size()[1] >= max_length:
                    break
                next_token_str = tokenizer.convert_ids_to_tokens(next_tokens[0].item())
                total_str += next_token_str
                if next_token_str == ACT:
                    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                    now_type = "Action"
                    span = ""
                elif next_token_str == TOPIC:
                    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                    now_type = "Topic"
                    span = ""
                else:
                    span = span + next_token_str
                    if now_type == "Action":
                        # set-based search for an action
                        match_str = set_search(span, action_list)
                        if  match_str is not None:
                            total_str += match_str[1:]
                            match_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(match_str.replace(" ", "").lower()))
                            match_ids = torch.LongTensor(match_ids).unsqueeze(0).to(input_ids.device)
                            input_ids = torch.cat([input_ids, match_ids], dim=-1)
                            now_type = "Topic"
                            span = ""
                        else:
                            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                    else:
                        # set-based search for a topic
                        match_str = set_search(span, topic_list)
                        if  match_str is not None:
                            total_str += match_str[1:]
                            match_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(match_str.replace(" ", "").lower()))
                            match_ids = torch.LongTensor(match_ids).unsqueeze(0).to(input_ids.device)
                            input_ids = torch.cat([input_ids, match_ids], dim=-1)
                            now_type = "Action"
                            span = ""
                        else:
                            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            return total_str
        else:
            while True:
                inputs["plans"][0], inputs["plans"][3] = input_ids, None
                # compute logits
                model_out = self(inputs, is_test=True)
                next_token_logits = model_out["lm_scores"][:, -1, :]
                next_tokens_scores = logits_processor(input_ids, next_token_logits)
                next_tokens = torch.argmax(next_tokens_scores, dim=-1)
                if eos_token_id is not None:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if eos_token_id is not None:
                    unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
                if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                    break
            return input_ids


# ============ Some utility functions ============

def get_logits_processor(
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    bad_words_ids: List[List[int]],
    min_length: int,
    max_length: int,
    eos_token_id: int,
    forced_bos_token_id: int,
    forced_eos_token_id: int,
    prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
    num_beams: int,
    num_beam_groups: int,
    diversity_penalty: float,
    remove_invalid_values: bool,
) -> LogitsProcessorList:
    """
    This mathod returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
    :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
    """
    processors = LogitsProcessorList()

    if diversity_penalty is not None and diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
            )
        )
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
    if min_length is not None and eos_token_id is not None and min_length > -1:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if prefix_allowed_tokens_fn is not None:
        processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams // num_beam_groups))
    if forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
    if forced_eos_token_id is not None:
        processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
    if remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    return processors

def get_stopping_criteria(max_length: Optional[int], max_time: Optional[float]) -> StoppingCriteriaList:
    stopping_criteria = StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    if max_time is not None:
        stopping_criteria.append(MaxTimeCriteria(max_time=max_time))
    return stopping_criteria
    
def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = copy.deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        print ("You set different `max_length` for stopping criteria and `max_length` parameter", flush=True)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria
