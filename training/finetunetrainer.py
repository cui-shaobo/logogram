<<<<<<< HEAD
"""
@Project  : abbreviation
@File     : finetunetrainer.py
@Date     : 09.07.23 15:12
"""

=======
>>>>>>> e57ad9ef1c7715764fb711c033c22b5715314d39
# usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import time

from rouge import Rouge
from nltk import bleu
import torch
import numpy as np
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import Adafactor
from transformers import T5Tokenizer
from accelerate import Accelerator

from dataset.dataset import get_loader
from model.t5 import FinetuneModel
from utils.utils import cal_running_avg_loss, progress_bar, eta


class FinetuneTrainer(object):
    def __init__(self, args):
        super(FinetuneTrainer, self).__init__()
        self.accelerator = Accelerator()

        self.args = args
        self.train_jsonl_file = args.train_jsonl_file
        self.val_jsonl_file = args.val_jsonl_file
        self.model_mode = args.model_mode

        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.train_num_epochs = args.train_epochs

        # Build dataloader.
        self.train_dataloader = None
        self.val_dataloader = None
        self.build_dataloader(args)


        # Build model.
        self.model = FinetuneModel(self.args)
        self.tokenizer = T5Tokenizer.from_pretrained(args.t5_model)

        # Save model.
        self.model_dir = args.model_dir

        params = self.model.parameters()
        self.optimizer = Adafactor(params)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)

        self.model, self.optimizer, self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader
        )

    def build_dataloader(self, args):
        self.train_dataloader = get_loader(self.train_jsonl_file, self.model_mode, self.train_batch_size, args.t5_model, args.max_length,
                                           args.max_decode_step)
        self.val_dataloader = get_loader(self.val_jsonl_file, self.model_mode, self.val_batch_size, args.t5_model, args.max_length,
                                         args.max_decode_step)

    def train(self):
        best_loss = 1e9
        avg_loss = 0.0
        best_metrics = 0
        start = time.time()
        for epoch in range(1, self.train_num_epochs + 1):
            self.model.train()
            batch_nb = len(self.train_dataloader)
            step = 1
            self.model.zero_grad()

            for batch_idx, batch in enumerate(self.train_dataloader, start=1):
                input_ids = batch['input_ids']
                labels = batch['labels']
                out = self.model(encoder_inputs=input_ids,
                                 decoder_labels=labels)
                loss = out.loss
                self.accelerator.backward(loss)
                self.optimizer.step()

                # compute running average
                avg_loss = cal_running_avg_loss(loss.mean(), avg_loss)
                msg = "Epoch: {} \t\t{}/{} {} - ETA : {}. loss: {}".format(
                    epoch, batch_idx, batch_nb,
                    progress_bar(batch_idx, batch_nb),
                    eta(start, batch_idx, batch_nb),
                    avg_loss)
                self.accelerator.print(msg)
                self.accelerator.print(msg, end="\r")
                step += 1
            self.scheduler.step()
            # For validation.
            val_loss = self.validation()
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model_with_val_loss(epoch=epoch, val_loss=best_loss)

        self.save_model(self.train_num_epochs)

    def save_model(self, epoch):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        ckpt = {
            "args": self.args,
            "state_dict": model_to_save.t5_model.state_dict()
        }
        model_save_path = os.path.join(self.model_dir, "{}_{}.pt".format(self.args.model_mode, epoch))
        torch.save(ckpt, model_save_path)

    def save_model_with_val_loss(self, epoch, val_loss):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        ckpt = {
            "args": self.args,
            "state_dict": model_to_save.t5_model.state_dict()
        }
        model_save_path = os.path.join(self.model_dir,
                                       "{}_epoch-{}_loss-{}.pt".format(self.args.model_mode, epoch, val_loss))
        torch.save(ckpt, model_save_path)

    def validation(self):
        val_losses = []
        with torch.no_grad():
            self.model.eval()
            for batch_idx, batch in enumerate(self.val_dataloader, start=1):
                input_ids = batch['input_ids']
                output_ids = batch['output_ids']
                loss = self.model(input_ids=input_ids,
                                  decoder_labels=output_ids)
                val_losses.append(loss.cpu())
        return np.mean(val_losses)

