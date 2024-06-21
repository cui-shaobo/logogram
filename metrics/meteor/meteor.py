# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
import subprocess
import pkg_resources

METEOR_JAR = "./meteor-1.5.jar"


class Meteor(object):

    def __init__(self, language="en", norm=True):
        self.meteor_cmd = ["java", "-jar", "-Xmx2G", METEOR_JAR,
                           "-", "-", "-stdio", "-l", language]

        if norm:
            self.meteor_cmd.append("-norm")

        self.meteor_p = subprocess.Popen(
            self.meteor_cmd, stdin=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, bufsize=1)

        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        imgIds = sorted(list(gts.keys()))
        scores = []
        eval_line = "EVAL"
        self.lock.acquire()

        for i in imgIds:
            assert(len(res[i]) == 1)

            hypothesis_str = res[i][0].replace("|||", "").replace("  ", " ")
            score_line = " ||| ".join(
                ("SCORE", " ||| ".join(gts[i]), hypothesis_str))
            score_line = score_line.replace('\n', '').replace('\r', '')

            # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
            self.meteor_p.stdin.write(score_line + "\n")
            stat = self.meteor_p.stdout.readline().strip()
            eval_line += " ||| {}".format(stat)

        # Send to METEOR
        self.meteor_p.stdin.write(eval_line + "\n")
        # print('eval_line: {}'.format(eval_line))
        # Collect segment scores
        for i in range(len(imgIds)):
            score = float(self.meteor_p.stdout.readline().strip())
            scores.append(score)

        # Final score
        final_score = 100 * float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return final_score, scores

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.wait()
        self.lock.release()

    @staticmethod
    def method():
        return "METEOR"
