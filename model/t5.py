# usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer


class FinetuneModel(nn.Module):
    def __init__(self, args):
        super(FinetuneModel, self).__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
        self.model_mode = args.model_mode

    # A fake wrapper for T5Model. To be modified by further incorporation of attention and other modules.
    def forward(self, encoder_inputs, decoder_labels):
        outputs = self.t5_model(input_ids=encoder_inputs,
                                labels=decoder_labels
                                )
        return outputs