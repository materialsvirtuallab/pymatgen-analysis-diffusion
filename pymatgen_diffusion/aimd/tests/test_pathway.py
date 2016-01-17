# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

from __future__ import division, unicode_literals, print_function

__author__ = "Iek-Heng Chu"
__version__ = 1.0
__date__ = "01/16"


import unittest
import os

import numpy as np
from pymatgen_diffusion.aimd.pathway import ProbabilityDensityAnalysis
from pymatgen import Structure

tests_dir = os.path.dirname(os.path.abspath(__file__))

class ProbabilityDensityTest(unittest.TestCase):

    def test_probability(self):
        traj_file = os.path.join(tests_dir, "cNa3PS4_trajectories.npy")
        struc_file = os.path.join(tests_dir, "cNa3PS4.cif")

        trajectories = np.load(traj_file)
        structure = Structure.from_file(struc_file, False)

        #ProbabilityDensityAnalysis object
        pda = ProbabilityDensityAnalysis(structure, trajectories, interval=0.5)
        dV = pda.structure.lattice.volume / pda.lens[0] / pda.lens[1] / pda.lens[2]
        Pr_tot = np.sum(pda.Pr) * dV

        self.assertAlmostEqual(pda.Pr.max(), 0.030735573102, 12)
        self.assertAlmostEqual(pda.Pr.min(), 0.0, 12)
        self.assertAlmostEqual(Pr_tot, 1.0, 12)

if __name__ == "__main__":
    unittest.main()
