# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import evaluate


class Rouge(object):
    """
    Class for computing ROUGE-L score for a set of 
    candidate sentences for the MS COCO test set
    """

    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """

        assert(len(candidate) == 1)
        assert(len(refs) > 0)
        
        rouge = evaluate.load("rouge")
        # print(candidate, refs)
        results = rouge.compute(predictions=candidate, references=refs)
        # print("1")
        
        return results["rougeL"]

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and 
        candidate sentences for the dataset.
        :param gts: dict : ground_truth
        :param res: dict : results of predict
        :returns: average_score: float (mean ROUGE-L score)
        """
        from tqdm import tqdm
        print("Computing ROUGE-L score...")

        score = []

        for idx in tqdm(sorted(gts.keys())):
            hypo = res[idx]
            ref = gts[idx]
            score.append(self.calc_score(hypo, ref))

            # Sanity check
            assert(isinstance(hypo, list))
            assert(isinstance(ref, list))
            assert(len(hypo) == 1)
            assert(len(ref) > 0)

        average_score = np.mean(np.array(score))

        # convert to percentage
        return 100 * average_score, np.array(score)

    @staticmethod
    def method():
        return "ROUGE-L"