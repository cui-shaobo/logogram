
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from transformers import AutoTokenizer


class WordLikeness(object):
    """
    Class for computing WordLikeness score for generated shorthands.
    Note that this evaluation metric doesn't use the ground truth for evaluation.
    Besides, WordLikeness is case-insensitive.
    """

    def __init__(self, tokenizer_type='t5-base'):
        self.beta = 1.2
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, model_max_length=512)
        self.tokenizer_type = tokenizer_type
        assert self.tokenizer_type in ['t5-base', 'bert-base-uncased']


    def compute_score(self, gts, res):

        assert len(gts.keys()) == len(res.keys())

        scores = []
        for idx in gts.keys():
            hypo = "".join(res[idx][0].split()).lower()
            if hypo == "":
                score = 0
            else:
                if self.tokenizer_type == 't5-base':                # SentencePiece tokenizer
                    score = 1 - (len([e for e in self.tokenizer.tokenize(hypo) if e != '▁'])-1) / len(hypo)
                elif self.tokenizer_type == 'bert-base-uncased':    # WordPiece tokenizer
                    score = 1 - (len(self.tokenizer.tokenize(hypo))-1) / len(hypo)
            scores.append(score)

        return sum(scores)/len(scores), scores


    @staticmethod
    def method():
        return "WordLikeness"