# -*- coding: utf-8 -*-

import argparse
import random

import torch
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm

import numpy as np
from transformers import T5Tokenizer
from accelerate.utils import set_seed
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from reward.rl_trainer import RLTrainer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler

def build_dataset(
    config, dataset_name="allenai/real-toxicity-prompts", input_min_text_length=5, input_max_text_length=10):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, split="train")

    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity > 0.3

    ds = ds.filter(filter_fn, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]

        sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

    return ds


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="ybelkada/gpt-j-6b-sharded-bf16", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="./gpt-j-6B-detoxified-long-context-26-shl-1e4-final",
        metadata={"help": "the path to save the model"},
    )


def get_args():
    set_seed(42)

    parser = argparse.ArgumentParser(description='RL')

    # dataset
    parser.add_argument(
        '--train_jsonl_file', type=str, default="./data/single_word_with_replacement.jsonl",
        help="train file for transformer")
    parser.add_argument(
        '--val_jsonl_file', type=str, default="./data/single_word_with_replacement.jsonl",
        help="val file for transformer")

    # training
    parser.add_argument(
        "--t5_model", type=str, default="t5-base")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="total batch size")
    parser.add_argument(
        "--val_batch_size", type=int, default=16, help="total batch size")
    parser.add_argument(
        "--max_length", type=int, default=128, help="max length for input document")
    parser.add_argument(
        "--max_decode_step", type=int, default=64, help="maximum decode step")
    parser.add_argument(
        '--train_epochs', type=int, default=10, help='Number of epochs to train')

    parser.add_argument("--random_seed", default=1004, type=int, help="Random state(seed)")

    # Model finetune for different input and output.
    parser.add_argument("--model_mode", type=str, choices=['abstract2description',
                                                           'abstract-description2shorthand'],
                        default='abstract2description')

    parser.add_argument("--use-cuda", type=bool, default=True)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    config = PPOConfig(
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        ppo_epochs=100,
        mini_batch_size=script_args.mini_batch_size,
        batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    )

    # We retrieve the dataloader by calling the `build_dataset` function.
    min_input_length = 30
    max_input_length = 40
    dataset = build_dataset(config, input_min_text_length=min_input_length, input_max_text_length=max_input_length)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)


    # Now let's build the model, the reference model, and the tokenizer. We first load the model
    # in bfloat16 to save memory using `transformers`.
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
    # And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    # We create a reference model by sharing 20 layers
    ref_model = create_reference_model(model, num_shared_layers=20)



    # We make sure to use `Adam` optimizer on the model parameters that require gradients.
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    # GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # We then build the reward pipeline, we will use the toxicity model to compute the reward.
    # We first load the toxicity model and tokenizer.
    toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
    toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
    # We load the toxicity model in fp16 to save memory.
    toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16).to(
        ppo_trainer.accelerator.device
    )

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    output_min_length = 20
    output_max_length = 30
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    model_save_path = script_args.model_save_path

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response from the policy model
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Compute sentiment score # noqa
        texts = batch["response"]
        toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            ppo_trainer.accelerator.device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()

        rewards = [torch.tensor(output) for output in toxicity_labels]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        # Save model every 100 epochs
        if epoch % 100 == 0:
            if ppo_trainer.accelerator.is_main_process:
                ppo_trainer.save_pretrained(model_save_path)


if __name__ == '__main__':
    main()