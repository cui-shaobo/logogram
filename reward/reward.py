# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from transformers import AutoTokenizer
from metrics.wordlikeness.wordlikeness import WordLikeness
from metrics.lcsratio.lcsratio import LCSRatio
from metrics.wordcoverage.wordcoverage import WordCoverage


class Reward():

    def __init__(self,
                 tokenizer_type='t5-base',
                 wordlikeliness=True,
                 lcsratio=True,
                 wordcoverage=True
                 ):
        self.beta = 1.2
        self.tokenizer_type = tokenizer_type
        self.wordlikeness = wordlikeliness
        self.lcsratio = lcsratio
        self.wordcoverage = wordcoverage

    def _harmonic_score(self, description: str, shorthand: str):
        scores = []
        if self.wordlikeness:
            wordlikeness = WordLikeness(self.tokenizer_type)
            scores.append(wordlikeness.compute_score({0: [description]}, {0: [shorthand]})[0])
        if self.lcsratio:
            lcsratio = LCSRatio()
            scores.append(lcsratio.compute_score({0: [description]}, {0: [shorthand]})[0])
        if self.wordcoverage:
            wordcoverage = WordCoverage()
            scores.append(wordcoverage.compute_score({0: [description]}, {0: [shorthand]})[0])

        assert len(scores) > 0, "At least one metric should be used for reward calculation"
        # Return the harmonic mean of all scores
        if 0 in scores:
            return 0
        else:
            return len(scores) / sum([1 / score for score in scores])

    def get_reward(self, description: str, shorthand: str):
        shorthand = shorthand.lower()
        if shorthand == "" or description == "":
            return -100
        else:
            return self._harmonic_score(description, shorthand)