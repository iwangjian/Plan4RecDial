# -*- coding: utf-8 -*-
import argparse
import os
import sys
import logging
import json
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from model.TCP import TCP
from utils.dataset import DuRecDialDataset
from utils.data_utils import get_tokenizer, combine_tokens
from utils.data_utils import get_action_set, get_topic_set, get_eval_output
from utils.data_collator import PlanCollator
from utils.trainer import Trainer

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  handlers=[
      logging.StreamHandler(sys.stdout)
  ]
)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--random_seed', type=int, default=42)
    
    # ==================== Data ====================
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--dev_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--bert_dir', type=str, default="config/bert-base-chinese")
    parser.add_argument('--use_knowledge_hop', type=str2bool, default="False")
    parser.add_argument('--cache_dir', type=str, default="caches")
    parser.add_argument('--log_dir', type=str, default="logs")
    
    # ==================== Train ====================
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--log_steps', type=int, default=400)
    parser.add_argument('--validate_steps', type=int, default=2000)
    parser.add_argument('--max_seq_len', type=int, default=512) 
    parser.add_argument('--max_plan_len', type=int, default=256)
    parser.add_argument('--max_memory_hop', type=int, default=3)
    parser.add_argument('--turn_type_size', type=int, default=16)
    parser.add_argument('--use_gpu', type=str2bool, default="True")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warm_up_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--ff_embed_dim', type=int, default=3072)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--decoder_layerdrop', type=float, default=0.1)
    parser.add_argument('--max_position_embeddings', type=int, default=512)
    parser.add_argument('--share_decoder_embedding', type=str2bool, default="False")
    parser.add_argument('--scale_embedding', type=str2bool, default="True")
    parser.add_argument('--init_std', type=float, default=0.02)
    parser.add_argument('--decoder_attention_heads', type=int, default=8)
    parser.add_argument('--decoder_layers', type=int, default=12)
    parser.add_argument('--output_attentions', type=bool, default=False)
    parser.add_argument('--output_hidden_states', type=bool, default=False)
    parser.add_argument('--use_cache', type=bool, default=False)
    parser.add_argument('--activation_function', type=str, default="gelu")
    parser.add_argument('--decoder_ffn_dim', type=int, default=3072)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation_dropout', type=float, default=0.1)
    parser.add_argument('--attention_dropout', type=float, default=0.1)

    #==================== Generate ====================
    parser.add_argument('--infer_checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--decoding_mode', type=str, default="greedy", choices=["greedy", "topk", "topp"])
    parser.add_argument('--use_ssd', type=str2bool, default="False")
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--bad_words_ids', type=list, default=None)
    parser.add_argument('--min_length', type=int, default=0)
    parser.add_argument('--diversity_penalty', type=float, default=0.0)
    parser.add_argument('--output_scores', type=bool, default=False)
    parser.add_argument('--return_dict_in_generate', type=bool, default=False)
    parser.add_argument('--remove_invalid_values', type=bool, default=False)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0.9)
   
    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def print_args(args):
    print("=============== Args ===============")
    for k in vars(args):
        print("%s: %s" % (k, vars(args)[k]))

def set_seed(args):
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

def run_train(args):
    logging.info("=============== Training ===============")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer, num_added_tokens, token_id_dict = get_tokenizer(config_dir=args.bert_dir)
    args.vocab_size = len(tokenizer)
    args.pad_token_id = token_id_dict["pad_token_id"]
    args.bos_token_id = token_id_dict["bos_token_id"]
    args.eos_token_id = token_id_dict["eos_token_id"]
    logging.info("{}: Add {} additional special tokens.".format(type(tokenizer).__name__, num_added_tokens))

    # define dataset
    train_dataset = DuRecDialDataset(data_path=args.train_data, tokenizer=tokenizer, data_partition='train',\
        cache_dir=args.cache_dir,  max_seq_len=args.max_seq_len, max_plan_len=args.max_plan_len,\
        turn_type_size=args.turn_type_size, use_knowledge_hop=args.use_knowledge_hop)
    dev_dataset = DuRecDialDataset(data_path=args.dev_data, tokenizer=tokenizer, data_partition='dev',\
        cache_dir=args.cache_dir,  max_seq_len=args.max_seq_len, max_plan_len=args.max_plan_len,\
        turn_type_size=args.turn_type_size, use_knowledge_hop=args.use_knowledge_hop)
    
    # create dataloader
    collator = PlanCollator(device=device, padding_idx=args.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator.custom_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.custom_collate)

    # build model
    if args.load_checkpoint is not None:
        model = torch.load(args.load_checkpoint)
    else:
        model = TCP(args=args)
    model.to(device)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total parameters: {}\tTrainable parameters: {}".format(total_num, trainable_num))
    
    # build trainer and execute model training
    trainer = Trainer(model=model, train_loader=train_loader, dev_loader=dev_loader,
        log_dir=args.log_dir, log_steps=args.log_steps, validate_steps=args.validate_steps, 
        num_epochs=args.num_epochs, lr=args.lr, warm_up_ratio=args.warm_up_ratio,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm
    )
    trainer.train()


def run_test(args):
    logging.info("=============== Testing ===============")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer, _, token_id_dict = get_tokenizer(config_dir=args.bert_dir)
    args.pad_token_id = token_id_dict["pad_token_id"]

    if args.use_ssd:
        if args.test_batch_size != 1:
            raise ValueError("Error! If you use set-search dcoding, `test_batch_size` must be 1.")
        data_dir = '/'.join(str(args.test_data).split('/')[:-1])
        action_set = get_action_set(data_dir)
        topic_sets = get_topic_set(args.test_data)       

    test_dataset = DuRecDialDataset(data_path=args.test_data, tokenizer=tokenizer, data_partition="test", 
        cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, max_plan_len=args.max_plan_len, 
        turn_type_size=args.turn_type_size, use_knowledge_hop=args.use_knowledge_hop,
        is_test=True)
    collator = PlanCollator(device=device, padding_idx=args.pad_token_id)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collator.custom_collate)

    if args.use_ssd:
        assert len(topic_sets) == len(test_loader), "`test_batch_size` must be 1."

    if args.infer_checkpoint is not None:
        model_path = os.path.join(args.log_dir, args.infer_checkpoint)
    else:
        model_path = os.path.join(args.log_dir, "best_model.bin")
    model = torch.load(model_path)
    logging.info("Model loaded from [{}]".format(model_path))
    model.to(device)
    model.eval()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_prefix = model_path.split('/')[-1].replace(".bin", "_test.txt")
    output_path = os.path.join(args.output_dir, output_prefix)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, inputs in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                if args.use_ssd:
                    return_str = model.generate(args, inputs, action_set, topic_sets[idx], tokenizer)
                    sentences = [return_str]
                else:
                    output = model.generate(args, inputs)
                    sentences = combine_tokens(output, tokenizer)
                for path_str in sentences:
                    # process reverse path
                    action, topic = get_eval_output(path_str)
                    plan = {"action": action, "topic": topic, "plan": path_str}
                    line = json.dumps(plan, ensure_ascii=False)
                    f.write(line + "\n")
                    f.flush()
    logging.info("Saved output to [{}]".format(output_path))


if __name__ == "__main__":
    args = parse_config()
    set_seed(args)
    
    if args.mode == "train":
        print_args(args)
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    else:
        exit("Please specify the \"mode\" parameter!")
