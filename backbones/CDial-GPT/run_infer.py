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
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BertTokenizer, OpenAIGPTLMHeadModel
from data_utils import load_data
from dataset_cdgpt import NEW_ADD_TOKENS, CDialGPTDataset


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


def sample_sequence(model, context, tokenizer, args):
    special_tokens_ids = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    context = torch.tensor(context, dtype=torch.long, device=args.device).unsqueeze(0)
    generated = context
    n_ctx = model.config.n_ctx
    output_ids = []

    for i in range(args.max_dec_len):
        input_ids = generated[0][-(n_ctx - 1):].unsqueeze(0)

        lm_output = model(input_ids, return_dict=True)
        logits = lm_output["logits"]
        logits = logits[0, -1, :] / args.temperature
        if args.top_k > 0 or (args.top_p > 0 and args.top_p <= 1):
            filtered_logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.topk(probs, 1)[1]
        
        if i < args.min_dec_len and next_token.item() in special_tokens_ids:
            while next_token.item() in special_tokens_ids:
                next_token = torch.multinomial(probs, num_samples=1)
        output_ids.append(next_token.item())
        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

        if next_token.item() in special_tokens_ids:
            break
    
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    output_text = output_text.replace(" ", "")
    return output_text
    
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
    parser.add_argument("--output_dir", type=str, default="", help="Dir for storing generated output")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max_seq_len", type=int, default=432, help="Maximum sequence length")
    parser.add_argument("--max_dec_len", type=int, default=80, help="Maximum length of the output utterances")
    parser.add_argument("--min_dec_len", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.0, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
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

    model_dir = args.model_dir + '_w_TCP' if args.use_tcp else args.model_dir
    
    logger.info("Loading tokenizer from [{}]".format(model_dir))
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    special_tokens_dict = {'additional_special_tokens': NEW_ADD_TOKENS}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    logger.info("Loading model from [{}]".format(model_dir))
    model = OpenAIGPTLMHeadModel.from_pretrained(model_dir)
    model.to(args.device)
    model.eval()

    # prepare data
    test_data = load_data(tokenizer, logger, args.test_path, args.cache_dir, 
        data_partition="test", use_tcp=args.use_tcp, tcp_path=args.tcp_path)
    
    logger.info("Evaluating...")
    eval_dataset = CDialGPTDataset(test_data, tokenizer, max_seq_len=args.max_seq_len, max_tgt_len=args.max_dec_len, lm_labels=True)
    eval_loader = DataLoader(eval_dataset, collate_fn=eval_dataset.collate, batch_size=1, shuffle=False)
    losses = []
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
            lm_output = model(input_ids, labels=lm_labels, return_dict=True)
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
    test_dataset = CDialGPTDataset(test_data, tokenizer, max_seq_len=args.max_seq_len, lm_labels=False)
    
    with open(output_path, 'w', encoding="utf-8") as f:
        # generate responses
        with torch.no_grad():
            for instance in tqdm(test_dataset, mininterval=1):
                # Only work for batch size 1 for now
                history = instance["input_ids"]
                output_text = sample_sequence(model, history, tokenizer, args)
                sample = {"response": output_text}
                line = json.dumps(sample, ensure_ascii=False)
                f.write(line + "\n")
                f.flush()
    logger.info("Saved output to [{}]".format(output_path))


if __name__ == "__main__":
    main()
