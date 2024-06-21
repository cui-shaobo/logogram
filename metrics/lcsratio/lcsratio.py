#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class LCSRatio(object):
    """
    Class for computing LCARatio score for generated shorthands.
    Note that this evaluation metric doesn't use the ground truth for evaluation.
    Besides, LCARatio is case-insensitive.
    """

    def __init__(self, tokenizer_type='t5-base'):
        pass

    def _lcs(self, X:str, Y:str):
        # get lcs string between X and Y
        m = len(X)
        n = len(Y)
        # An (m+1) times (n+1) matrix
        C = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Since Python sequences start with 0,
                # but we start with 1, we need to -1
                if X[i - 1] == Y[j - 1]:
                    C[i][j] = C[i - 1][j - 1] + 1
                else:
                    C[i][j] = max(C[i][j - 1], C[i - 1][j])

        # C[m][n] contains the length of LCS
        # and the actual LCS can be computed
        # using backtracking
        # Start from the right-most-bottom-most corner and
        # one by one store characters in lcs[]
        lcs = ""
        i = m
        j = n

        while i > 0 and j > 0:
            # If current character in X[] and Y are same, then
            # current character is part of LCS
            if X[i - 1] == Y[j - 1]:
                lcs = X[i - 1] + lcs
                i -= 1
                j -= 1

            # If not same, then find the larger of two and
            # go in the direction of larger value
            elif C[i - 1][j] > C[i][j - 1]:
                i -= 1
            else:
                j -= 1

        return lcs     

    def compute_score(self, gts, res):

        assert len(gts.keys()) == len(res.keys())

        scores = []
        for idx in gts.keys():
            shorthand = "".join(res[idx][0].split()).lower()
            description = gts[idx][0].lower()
            if shorthand == "" or description == "":
                score = 0
            else:
                lcs = self._lcs(shorthand, description)
                score = len(lcs) / len(shorthand)
            scores.append(score)

        return sum(scores)/len(scores), scores


    @staticmethod
    def method():
        return "LCSRatio"