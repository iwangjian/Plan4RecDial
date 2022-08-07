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
from transformers import BertTokenizer, BartForConditionalGeneration
from data_utils import load_data, NEW_ADD_TOKENS
from dataset_bart import BartDataset


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
    parser.add_argument("--test_batch_size", type=int, default=2, help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_dec_len", type=int, default=80, help="Maximum length of the output utterances")
    parser.add_argument("--min_dec_len", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling softmax temperature")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam size")
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
    model = BartForConditionalGeneration.from_pretrained(model_dir)
    model.to(args.device)
    model.eval()

    # prepare data
    dataset = load_data(tokenizer, logger, args.test_path, args.cache_dir, 
        data_partition="test", use_tcp=args.use_tcp, tcp_path=args.tcp_path)

    test_dataset = BartDataset(dataset, tokenizer, max_seq_len=args.max_seq_len)
    
    logger.info("Evaluating...")
    eval_loader = DataLoader(test_dataset, collate_fn=test_dataset.collate, batch_size=1, shuffle=False)
    losses = []
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids, decoder_input_ids, labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
            lm_output = model(
                input_ids=input_ids, 
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
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
    
    test_loader = DataLoader(test_dataset, collate_fn=test_dataset.collate, batch_size=args.test_batch_size, shuffle=False)
    logger.info("Generating...")
    with open(output_path, 'w', encoding="utf-8") as f:
        # generate responses
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids, _, _ = tuple(input_tensor.to(args.device) for input_tensor in batch)
                outputs = model.generate(input_ids,
                    max_length=args.max_dec_len,
                    min_length=args.min_dec_len,
                    num_beams=args.num_beams,
                    early_stopping=True
                )
                output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                output_texts = ["".join(output.split()) for output in output_tokens]
                for out_text in output_texts:
                    sample = {"response": out_text}
                    line = json.dumps(sample, ensure_ascii=False)
                    f.write(line + "\n")
                    f.flush()
    logger.info("Saved output to [{}]".format(output_path))


if __name__ == "__main__":
    main()
