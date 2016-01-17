# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

from __future__ import division, unicode_literals, print_function

__author__ = "Iek-Heng Chu"
__version__ = 1.0
__date__ = "01/16"

import unittest
import os
import json

import numpy as np
from pymatgen_diffusion.aimd.van_hove import VanHoveAnalysis
from pymatgen.analysis.diffusion_analyzer import DiffusionAnalyzer

tests_dir = os.path.dirname(os.path.abspath(__file__))


class VanHoveTest(unittest.TestCase):
    def test_van_hove(self):
        data_file = os.path.join(tests_dir, "cNa3PS4_pda.json")
        data = json.load(open(data_file, "r"))
        obj = DiffusionAnalyzer.from_dict(data)

        vh = VanHoveAnalysis(pmg_diff_analyzer=obj, avg_nsteps=5, ngrid=101, rmax=10.0,
                             step_skip=5, sigma=0.1, species=["Li", "Na"])

        check = np.shape(vh.gsrt) == (20, 101) and np.shape(vh.gdrt) == (20, 101)
        self.assertTrue(check)
        self.assertAlmostEqual(vh.gsrt[0, 0], 3.98942280401, 10)
        self.assertAlmostEqual(vh.gdrt[10, 0], 9.68574868168, 10)


if __name__ == "__main__":
    unittest.main()
