# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

from __future__ import division, unicode_literals, print_function
from pymatgen import Structure, Lattice

__author__ = "Iek-Heng Chu"
__version__ = 1.0
__date__ = "01/16"

import unittest
import os
import json

import numpy as np
import matplotlib

matplotlib.use("pdf")

from pymatgen_diffusion.aimd.van_hove import VanHoveAnalysis, RadialDistributionFunction, EvolutionAnalyzer
from pymatgen.analysis.diffusion_analyzer import DiffusionAnalyzer

tests_dir = os.path.dirname(os.path.abspath(__file__))


class VanHoveTest(unittest.TestCase):
    def test_van_hove(self):
        data_file = os.path.join(tests_dir, "cNa3PS4_pda.json")
        data = json.load(open(data_file, "r"))
        obj = DiffusionAnalyzer.from_dict(data)

        vh = VanHoveAnalysis(diffusion_analyzer=obj, avg_nsteps=5, ngrid=101, rmax=10.0,
                             step_skip=5, sigma=0.1, species=["Li", "Na"])

        check = np.shape(vh.gsrt) == (20, 101) and np.shape(vh.gdrt) == (20, 101)
        self.assertTrue(check)
        self.assertAlmostEqual(vh.gsrt[0, 0], 3.98942280401, 10)
        self.assertAlmostEqual(vh.gdrt[10, 0], 9.68574868168, 10)


class RDFTest(unittest.TestCase):
    def test_rdf(self):
        data_file = os.path.join(tests_dir, "cNa3PS4_pda.json")
        data = json.load(open(data_file, "r"))
        obj = DiffusionAnalyzer.from_dict(data)

        structure_list = []
        for i, s in enumerate(obj.get_drift_corrected_structures()):
            structure_list.append(s)
            if i == 9: break

        obj = RadialDistributionFunction(structures=structure_list, ngrid=101, rmax=10.0,
                                         cellrange=1, sigma=0.1, species=["Na", "P", "S"])

        check = np.shape(obj.rdf)[0] == 101 and np.argmax(obj.rdf) == 34
        self.assertTrue(check)
        self.assertAlmostEqual(obj.rdf.max(), 1.634448, 4)

    def test_rdf_coordination_number(self):
        # create a simple cubic lattice
        coords = np.array( [ [ 0.5, 0.5, 0.5 ] ] )
        atom_list = [ 'S' ]
        lattice = Lattice.from_parameters( a=1.0, b=1.0, c=1.0, alpha=90, beta=90, gamma=90 )
        structure = Structure( lattice, atom_list, coords )
        rdf = RadialDistributionFunction( structures=[ structure ], species=['S'], rmax=5.0, sigma=0.1, ngrid=500 )
        self.assertEqual( rdf.coordination_number[101], 6.0 )

    def test_rdf_two_species_coordination_number(self):
        # create a structure with interpenetrating simple cubic lattice
        coords = np.array( [ [ 0.0, 0.0, 0.0 ],
                             [ 0.5, 0.5, 0.5 ] ] )
        atom_list = [ 'S', 'Zn' ]
        lattice = Lattice.from_parameters( a=1.0, b=1.0, c=1.0, alpha=90, beta=90, gamma=90 )
        structure = Structure( lattice, atom_list, coords )
        rdf = RadialDistributionFunction( structures=[ structure ], species=['S'], reference_species=['Zn'], rmax=5.0, sigma=0.1, ngrid=500 )
        self.assertEqual( rdf.coordination_number[101], 8.0 )
 
class EvolutionAnalyzerTest(unittest.TestCase):
    def test_get_df(self):
        data_file = os.path.join(tests_dir, "cNa3PS4_pda.json")
        data = json.load(open(data_file, "r"))
        obj = DiffusionAnalyzer.from_dict(data)

        structure_list = []
        for i, s in enumerate(obj.get_drift_corrected_structures()):
            structure_list.append(s)
            if i == 9: break
        eva = EvolutionAnalyzer(structure_list, rmax=10, step=1, time_step=2)
        rdf = eva.get_df(EvolutionAnalyzer.rdf, pair=("Na", "Na"))
        atom_dist = eva.get_df(EvolutionAnalyzer.atom_dist, specie="Na", direction="c")
        check = np.shape(rdf) == (10, 101) and np.shape(atom_dist) == (10, 101) and eva.pairs[0] == ("Na", "Na")
        self.assertTrue(check)
        self.assertAlmostEqual(max(np.array(rdf)[0]), 1.772465, 4)

if __name__ == "__main__":
    unittest.main()
