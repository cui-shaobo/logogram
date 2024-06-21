import torch
import pandas as pd
from argparse import ArgumentParser
from dataclasses import dataclass, field
from t5_trainer import BruteT5PPOTrainer
from transformers import T5Tokenizer
from trl import AutoModelForSeq2SeqLMWithValueHead
from src.dataset import T5TitleDataset
import warnings


def parse_arguments():
    parser = ArgumentParser()

    # ModelArguments
    parser.add_argument("--train_data_path", default="./data/single_word_with_replacement_val.jsonl", help="training data file")
    parser.add_argument("--val_data_path", default="./data/single_word_with_replacement_val.jsonl", help="validation data file")
    parser.add_argument("--model_mode", default="abstract2description:shorthand",
                        help="model mode for training",
                        choices=["abstract2shorthand:description", "abstract2description:shorthand"])
    parser.add_argument("--shorthand_mode", default="tokenizer", help="how to tokenize shorthand for training",
                        choices=["tokenizer", "character"])
    parser.add_argument("--max_length", type=int, default=512, help="max token length for input texts")
    parser.add_argument("--max_decode_step", type=int, default=64, help="max token length for output texts")

    # CustomTrainingArguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="./models", help="Directory for saving model")
    parser.add_argument("--model_save_path", default="./models/t5-ppo", help="Directory for saving model")
    parser.add_argument("--ppo_epochs", type=int, default=100, help="Number of ppo training epochs")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Number of total saved model")
    parser.add_argument("--num_shared_layers", type=int, default=0, help="Number of shared layers between the active and the reference model")
    parser.add_argument("--learning_rate", type=float, default=1.41e-7, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Mini batch size for PPO")
    parser.add_argument("--init_kl_coef", type=float, default=0.002, help="Initial kl coefficient")
    parser.add_argument("--target_kl", type=float, default=0.001, help="Target kl")
    parser.add_argument("--ratio_threshold", type=float, default=10, help="Skip mini-batches with high PPO ratios that can cause loss spikes")
    parser.add_argument("--kl_penalty", type=str, default='kl', choices=['full', 'kl', 'abs', 'mse'],
                        help="kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl), 'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution")
    
    parser.add_argument("--log_with", type=str, default="wandb", choices=["wandb", "tensorboard"], help="Logging with wandb or tensorboard")

    return parser.parse_args()


if __name__ == "__main__":
    # Only print each warning once
    warnings.filterwarnings('once')
    
    args = parse_arguments()
    
    # Initialize the model and the tokenizer
    if args.model_mode == "abstract2shorthand:description" and args.shorthand_mode == "tokenizer":
        args.model_name = "models/t5-a2sd-token-base/checkpoint-5"
    elif args.model_mode == "abstract2shorthand:description" and args.shorthand_mode == "character":
        args.model_name = "models/t5-a2sd-char-base/checkpoint-5"
    elif args.model_mode == "abstract2description:shorthand" and args.shorthand_mode == "tokenizer":
        args.model_name = "models/t5-a2ds-token-base/checkpoint-5"
    elif args.model_mode == "abstract2description:shorthand" and args.shorthand_mode == "character":
        args.model_name = "models/t5-a2ds-char-base/checkpoint-5"
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.add_special_tokens({'sep_token': '</s>'})

    # Load datasets
    train_data = T5TitleDataset(args.train_data_path, args.model_mode, args.shorthand_mode, tokenizer, args.max_length, args.max_decode_step)
    val_data = T5TitleDataset(args.val_data_path, args.model_mode, args.shorthand_mode, tokenizer, args.max_length, args.max_decode_step)

    # Initiate the trainer
    trainer = BruteT5PPOTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer
    )
    
    # Train the model
    trainer.train()
