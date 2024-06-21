import torch
import pandas as pd
from argparse import ArgumentParser
from dataclasses import dataclass, field
from t5_inferencer import BruteT5Inferencer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.dataset import T5TitleDataset
import warnings


def parse_arguments():
    parser = ArgumentParser()

    # ModelArguments
    parser.add_argument("--model_name", default="t5-large", help="t5 model for training")
    parser.add_argument("--test_data_path", default="./data/single_word_with_replacement_test.jsonl", help="testing data file")
    parser.add_argument("--model_mode", default="abstract2shorthand:description",
                        help="model mode for training",
                        choices=["abstract2shorthand:description", "abstract2description:shorthand",
                                 "abstract2description", "abstract-description2shorthand"])
    parser.add_argument("--shorthand_mode", default="tokenizer", help="how to tokenize shorthand for training",
                        choices=["tokenizer", "character"])
    parser.add_argument("--max_length", type=int, default=512, help="max token length for input texts")
    parser.add_argument("--max_decode_step", type=int, default=64, help="max token length for output texts")

    # CustomInferenceArguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prediction_save_path", default="./prediction/brute_t5_predictions.csv", help="Directory for saving the prediction file")
    parser.add_argument("--per_device_test_batch_size", type=int, default=8, help="Batch size for testing")

    return parser.parse_args()


if __name__ == "__main__":
    # Only print each warning once
    warnings.filterwarnings('once')

    args = parse_arguments()
    
    # Initialize the model and the tokenizer
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, model_max_length=512)

    # Load datasets
    test_data = T5TitleDataset(args.test_data_path, args.model_mode, args.shorthand_mode, tokenizer, args.max_length, args.max_decode_step)

    # Initiate the trainer
    inferencer = BruteT5Inferencer(
        model=model,
        args=args,
        test_dataset=test_data,
        tokenizer=tokenizer
    )
    
    # Make predictions
    predictions, labels = inferencer.predict()
    
    # Convert predictions to text outputs and save to a CSV
    inferencer.save_predictions_to_csv(predictions, labels)
