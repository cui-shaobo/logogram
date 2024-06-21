import torch
import pandas as pd
from argparse import ArgumentParser
from dataclasses import dataclass, field
from t5_trainer import BruteT5Trainer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.dataset import T5TitleDataset
import warnings


def parse_arguments():
    parser = ArgumentParser()

    # ModelArguments
    parser.add_argument("--model_name", default="./pretrained_model/", help="t5 model for training")
    parser.add_argument("--train_data_path", default="./data/single_word_with_replacement_train.jsonl", help="training data file")
    parser.add_argument("--val_data_path", default="./data/single_word_with_replacement_val.jsonl", help="validation data file")
    parser.add_argument("--model_mode", default="abstract2shorthand:description",
                        help="model mode for training",
                        choices=["abstract2shorthand:description", "abstract2description:shorthand",
                                 "abstract2description", "abstract-description2shorthand"])
    parser.add_argument("--shorthand_mode", default="tokenizer", help="how to tokenize shorthand for training",
                        choices=["tokenizer", "character"])
    parser.add_argument("--max_length", type=int, default=512, help="max token length for input texts")
    parser.add_argument("--max_decode_step", type=int, default=64, help="max token length for output texts")

    # CustomTrainingArguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_save_path", default="./models/t5-brute-large", help="Directory for saving model")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Strength of weight decay")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Number of total saved model")

    return parser.parse_args()


if __name__ == "__main__":
    # Only print each warning once
    warnings.filterwarnings('once')
    
    args = parse_arguments()
    
    # Initialize the model and the tokenizer
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.add_special_tokens({'sep_token': '</s>'})

    # Load datasets
    train_data = T5TitleDataset(args.train_data_path, args.model_mode, args.shorthand_mode, tokenizer, args.max_length, args.max_decode_step)
    val_data = T5TitleDataset(args.val_data_path, args.model_mode, args.shorthand_mode, tokenizer, args.max_length, args.max_decode_step)

    # Initiate the trainer
    trainer = BruteT5Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer
    )
    
    # Train the model
    trainer.train()
