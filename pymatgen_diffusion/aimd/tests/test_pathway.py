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
from pymatgen_diffusion.aimd.pathway import ProbabilityDensityAnalysis, \
    SiteOccupancyAnalyzer
from pymatgen.analysis.diffusion_analyzer import DiffusionAnalyzer
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

    def test_probability_classmethod(self):
        file = os.path.join(tests_dir, "cNa3PS4_pda.json")
        data = json.load(open(file, "r"))
        diff_analyzer = DiffusionAnalyzer.from_dict(data)

        #ProbabilityDensityAnalysis object
        pda = ProbabilityDensityAnalysis.from_diffusion_analyzer(diffusion_analyzer=diff_analyzer,
                                                                 interval=0.5)
        dV = pda.structure.lattice.volume / pda.lens[0] / pda.lens[1] / pda.lens[2]
        Pr_tot = np.sum(pda.Pr) * dV

        self.assertAlmostEqual(pda.Pr.max(), 0.0361594977596, 8)
        self.assertAlmostEqual(pda.Pr.min(), 0.0, 12)
        self.assertAlmostEqual(Pr_tot, 1.0, 12)

class SiteOccupancyTest(unittest.TestCase):

    def test_site_occupancy(self):
        traj_file = os.path.join(tests_dir, "cNa3PS4_trajectories.npy")
        struc_file = os.path.join(tests_dir, "cNa3PS4.cif")

        trajectories = np.load(traj_file)
        structure = Structure.from_file(struc_file, False)

        coords_ref= [ss.frac_coords for ss in structure if ss.specie.symbol == "Na"]

        #SiteOccupancyAnalyzer object
        socc = SiteOccupancyAnalyzer(structure, coords_ref, trajectories,
                                     species=("Li", "Na"))
        site_occ = socc.site_occ
        self.assertAlmostEqual(np.sum(site_occ), len(coords_ref), 12)
        self.assertAlmostEqual(site_occ[11], 0.98, 12)
        self.assertAlmostEqual(site_occ[15], 0.875, 12)
        self.assertEqual(len(coords_ref), 48)

    def test_site_occupancy_classmethod(self):
        file = os.path.join(tests_dir, "cNa3PS4_pda.json")
        data = json.load(open(file, "r"))
        diff_analyzer = DiffusionAnalyzer.from_dict(data)

        structure = diff_analyzer.structure
        coords_ref = [ss.frac_coords for ss in structure if ss.specie.symbol == "Na"]

        #SiteOccupancyAnalyzer object
        socc = SiteOccupancyAnalyzer.from_diffusion_analyzer(coords_ref,
                                                            diffusion_analyzer=diff_analyzer)
        site_occ = socc.site_occ
        self.assertAlmostEqual(np.sum(site_occ), len(coords_ref), 12)
        self.assertAlmostEqual(site_occ[1], 0.98, 12)
        self.assertAlmostEqual(site_occ[26], 0.97, 12)
        self.assertEqual(len(coords_ref), 48)

if __name__ == "__main__":
    unittest.main()
