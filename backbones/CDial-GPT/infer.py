# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import logging
import random
import numpy as np
import argparse
from pprint import pformat
from itertools import chain
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import OpenAIGPTLMHeadModel, BertTokenizer

from utils.inputter_wb import USER, BOT, get_data
from utils.dataset_wb import SPECIAL_TOKENS, WBDataset


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def build_input_from_segments(history, reply, tokenizer, max_decode_length, with_eos=False, max_seq_length=512):
    """ Build a sequence of input from history and last reply """
    bos, _, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    
    if len(history[0]) > max_seq_length - max_decode_length - 2:
        history = history[0][-(max_seq_length-max_decode_length-2):]
    else:
        history = history[0]

    sequence = [[bos]] + [history] + [reply + ([eos] if with_eos else [])]
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
    return instance

def sample_sequence(history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(history, current_output, tokenizer, args.max_length, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], dtype=torch.long, device=args.device).unsqueeze(0)

        lm_output = model(input_ids, token_type_ids=token_type_ids, return_dict=True)
        logits = lm_output["logits"]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_tcp', type=str2bool, default="False", help="Whether or not use TCP-enhanced generation")
    parser.add_argument('--tcp_path', type=str, default=None, help="Path of the decoded plans by TCP.")
    parser.add_argument("--test_path", type=str, default=None, help="Path of the test dataset.")
    parser.add_argument("--cache_dir", type=str, default="dataset_cache",
                        help="Path or url of the dataset cache dir.")
    parser.add_argument("--model_dir", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--output_dir", type=str, default="", help="Dir for storing generated output.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.0,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_dir is None or args.model_dir == "":
        logging.error("Checkpoint needed!")
        return

    random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = BertTokenizer
    model_class = OpenAIGPTLMHeadModel
    model_dir = args.model_dir + '_w_TCP' if args.use_tcp else args.model_dir
    tokenizer = tokenizer_class.from_pretrained(model_dir, do_lower_case=True, never_split=[USER, BOT])
    model = model_class.from_pretrained(model_dir)

    model.to(args.device)
    model.eval()

    # prepare data
    dataset = get_data(tokenizer, logger, args.test_path, args.cache_dir, 
        data_partition="test", use_tcp=args.use_tcp, tcp_path=args.tcp_path)

    # evaluate
    test_dataset = WBDataset(dataset, tokenizer)
    test_loader = DataLoader(test_dataset, collate_fn=test_dataset.collate, batch_size=1, shuffle=False)
    logger.info("Evaluating...")
    losses = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
            lm_output = model(input_ids, labels=lm_labels, token_type_ids=token_type_ids, return_dict=True)
            lm_loss = lm_output["loss"]
            losses.append(float(lm_loss))
    avg_loss = np.mean(losses)
    logger.info("Avg loss: {}".format(avg_loss))
    logger.info("Avg ppl: {}".format(np.math.exp(avg_loss)))

    # set output dir
    output_dir = args.output_dir + '_w_TCP' if args.use_tcp else args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "output_test.txt")
    
    logger.info("Generating...")
    with open(output_path, 'w', encoding="utf-8") as f:
        # generate responses
        for instance in tqdm(dataset, mininterval=1):
            with torch.no_grad():
                history = instance[:-1]
                out_ids = sample_sequence(history, tokenizer, model, args)
                out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                sample = {"response": out_text.replace(" ", "")}
                line = json.dumps(sample, ensure_ascii=False)
                f.write(line + "\n")
                f.flush()
    logger.info("Saved output to [{}]".format(output_path))


if __name__ == "__main__":
    main()
