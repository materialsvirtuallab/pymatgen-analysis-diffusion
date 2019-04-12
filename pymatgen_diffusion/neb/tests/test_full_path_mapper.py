# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

from pymatgen_diffusion.neb.full_path_mapper import generic_groupby, FullPathMapper
from pymatgen.util.testing import PymatgenTest
import unittest
from pymatgen import Structure
import numpy as np
import os
import glob

__author__ = "Jimmy Shen"
__version__ = "1.0"
__date__ = "April 10, 2019"


class FullPathMapperTest(unittest.TestCase):
    def setUp(self):
        struct = Structure.from_file("./pathfinder_files/MnO2_full_Li.vasp")
        # Only one path in LiFePO4 with 4 A.
        self.fpm = FullPathMapper(structure=struct, migrating_specie='Li', max_path_length=4)

    def test_group_and_label_hops(self):
        """
        Check that the set of end points in a group of similiarly labeled hops are all the same
        """
        self.fpm.populate_edges_with_migration_paths()
        self.fpm.group_and_label_hops()
        edge_labs = np.array([d['hop_label'] for u, v, d in self.fpm.s_graph.graph.edges(data=True)])

        site_labs = np.array([(d['hop'].symm_structure.wyckoff_symbols[d['hop'].iindex], d['hop'].symm_structure.wyckoff_symbols[d['hop'].eindex]) for u, v, d in self.fpm.s_graph.graph.edges(data=True)])

        for itr in range(edge_labs.max()):
            sub_set = site_labs[edge_labs == itr]
            for end_point_labels in sub_set:
                self.assertTrue(sorted(end_point_labels)==sorted(sub_set[0]))

    if __name__ == '__main__':
        unittest.main()
