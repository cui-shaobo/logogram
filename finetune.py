# usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random

import torch
import numpy as np
from transformers import T5Tokenizer
from accelerate.utils import set_seed

from training.trainer import FinetuneTrainer


def main():

    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl_file", type=str,
                        default="./data/single_word_with_replacement.jsonl",
                        help="train file for transformer")
    parser.add_argument("--val_jsonl_file", type=str,
                        default="./data/single_word_with_replacement.jsonl",
                        help="val file for transformer")
    parser.add_argument("--t5_model", type=str, default="t5-base")
    parser.add_argument("--train_batch_size", help="total batch size",
                        type=int, default=16)
    parser.add_argument("--val_batch_size", help="total batch size",
                        type=int, default=16)
    parser.add_argument("--max_length", help="max length for input document",
                        default=128, type=int)
    parser.add_argument("--max_decode_step", type=int,
                        default=64, help="maximum decode step")
    parser.add_argument('--train_epochs',
                        help='Number of epochs to train',
                        type=int, default=10)

    # Model finetune for different input and output.
    parser.add_argument("--model_mode", type=str, choices=['abstract2description',
                                                           'abstract-description2shorthand'],
                        default='abstract2description')

    # contrastive learning
    parser.add_argument("--hidden_size", type=int, default=512)

    parser.add_argument("--pos_eps", type=float, default=3.0)
    parser.add_argument("--neg_eps", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--norm", type=int, default=2)

    parser.add_argument("--load", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="whether to activate debugging mode")
    parser.add_argument("--model_dir", type=str, default="./saved_model/")
    parser.add_argument("--model_path", type=str, default="")


    parser.add_argument("--rank", default=0,
                        help="The priority rank of current node.")
    parser.add_argument("--dist_backend", default="nccl",
                        help="Backend communication method. "
                             "NCCL is used for DistributedDataParallel")
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:8297",
                        help="DistributedDataParallel server")

    parser.add_argument("--distributed", action="store_true",
                        help="Use multiprocess distribution or not")
    parser.add_argument("--random_seed", default=1004, type=int,
                        help="Random state(seed)")
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--use-cuda", type=bool, default=True)

    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    args.tokenizer = tokenizer
    args.vocab_size = len(tokenizer.get_vocab())

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    trainer = FinetuneTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
