# usr/bin/env python
# -*- coding: utf-8 -*-

import json
import linecache
import os
import subprocess

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import T5Tokenizer


class T5Dataset(Dataset):
    def __init__(self, jsonl_file, model_mode, t5_model, max_length,
                 max_decode_step=64, debug=False):
        self.max_length = max_length
        self.max_decode_step = max_decode_step
        self.model_mode = model_mode

        self.tokenizer = T5Tokenizer.from_pretrained(t5_model)
        self.file_name = jsonl_file
        self.total_size = int(subprocess.check_output(
            "wc -l " + jsonl_file, shell=True).split()[0])
        if debug:
            self.total_size = 500

    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index + 1)
        json_dict = json.loads(line)

        if self.model_mode == "abstract2description":
            input_text = "Paper abstract: " + json_dict["Abstract"]
            output_text = "Paper title: " + json_dict["Title"]
        elif self.model_mode == "abstract-description2shorthand":
            input_text = "Paper abstract: " + json_dict["Abstract"] + "Title: " + json_dict["Title"]
            output_text = "Short title: " + json_dict["Abbreviation"]

        input_ids = self.tokenizer.encode(input_text,
                                          padding='max_length',
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors='pt'
                                          ).squeeze()
        # Add space to characters in shorthand.
        if self.model_mode == "abstract-description2shorthand":
            output_text = " ".join(list(output_text))

        labels = self.tokenizer.encode(output_text,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_decode_step,
                                       return_tensors='pt').squeeze()

        batch = {"input_ids": input_ids,
                 "labels": labels,
                 }
        return batch

    def __len__(self):
        return self.total_size


def get_loader(jsonl_file, model_mode, batch_size, t5_model, max_length, max_decode_step):
    dataset = T5Dataset(jsonl_file, model_mode, t5_model,
                        max_length, max_decode_step)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=True)
    return dataloader


def get_dist_loader(jsonl_file, batch_size, t5_model, max_length, max_decode_step):
    dataset = T5Dataset(
        jsonl_file,
        t5_model=t5_model,
        max_length=max_length,
        max_decode_step=max_decode_step
    )
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset=dataset,
                            sampler=sampler,
                            batch_size=batch_size,
                            drop_last=True)
    return dataloader
