# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

from pymatgen_diffusion.neb.full_path_mapper import FullPathMapper, ComputedEntryPath, get_all_sym_sites

from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.io.vasp import Chgcar
import unittest
from pymatgen import Structure
import numpy as np
from monty.serialization import loadfn
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

__author__ = "Jimmy Shen"
__version__ = "1.0"
__date__ = "April 10, 2019"


class FullPathMapperSimpleTest(unittest.TestCase):
    def setUp(self):
        struct = Structure.from_file(f"{dir_path}/full_path_files/MnO2_full_Li.vasp")
        self.fpm = FullPathMapper(
            structure=struct, migrating_specie='Li', max_path_length=4)

    def test_get_pos_and_migration_path(self):
        """
        Make sure that we can populate the graph with MigrationPath Objects
        """
        self.fpm._get_pos_and_migration_path(0, 1, 1)
        self.assertAlmostEqual(self.fpm.s_graph.graph[0][1][1]['hop'].length,
                               3.571248, 4)


class FullPathMapperComplexTest(unittest.TestCase):
    def setUp(self):
        struct = Structure.from_file(f"{dir_path}/full_path_files/MnO2_full_Li.vasp")
        self.fpm = FullPathMapper(
            structure=struct, migrating_specie='Li', max_path_length=4)
        self.fpm.populate_edges_with_migration_paths()
        self.fpm.group_and_label_hops()

    def test_group_and_label_hops(self):
        """
        Check that the set of end points in a group of similiarly labeled hops are all the same
        """
        edge_labs = np.array([
            d['hop_label']
            for u, v, d in self.fpm.s_graph.graph.edges(data=True)
        ])

        site_labs = np.array(
            [(d['hop'].symm_structure.wyckoff_symbols[d['hop'].iindex],
              d['hop'].symm_structure.wyckoff_symbols[d['hop'].eindex])
             for u, v, d in self.fpm.s_graph.graph.edges(data=True)])

        for itr in range(edge_labs.max()):
            sub_set = site_labs[edge_labs == itr]
            for end_point_labels in sub_set:
                self.assertTrue(sorted(end_point_labels) == sorted(sub_set[0]))

    def test_unique_hops_dict(self):
        """
        Check that the unique hops are inequilvalent
        """
        self.fpm.get_unique_hops_dict()
        unique_list = [v for k, v in self.fpm.unique_hops.items()]
        all_pairs = [(mg1, mg2) for i1, mg1 in enumerate(unique_list)
                     for mg2 in unique_list[i1 + 1:]]

        for migration_path in all_pairs:
            self.assertNotEqual(migration_path[0], migration_path[1])


class ComputedEntryPathTest(unittest.TestCase):
    def setUp(self):
        self.test_ents_MOF = loadfn(
            f'{dir_path}/full_path_files/Mn6O5F7_cat_migration.json')
        self.aeccar_MOF = Chgcar.from_file(
            f'{dir_path}/full_path_files/AECCAR_Mn6O5F7.vasp')
        self.cep = ComputedEntryPath(
            base_struct_entry=self.test_ents_MOF['ent_base'],
            migrating_specie='Li',
            single_cat_entries=self.test_ents_MOF['one_cation'],
            base_aeccar=self.aeccar_MOF)

        # structure matcher used for validation
        self.rough_sm = StructureMatcher(
            comparator=ElementComparator(),
            primitive_cell=False,
            ltol=0.5,
            stol=0.5,
            angle_tol=7)

    def test_get_all_sym_sites(self):
        """
        Inserting a Li into each one of the proposed structures should result in an equivalent structure
        (using sloppier tolerences)
        """
        for ent in self.cep.translated_single_cat_entries:
            li_sites = get_all_sym_sites(ent, self.cep.base_struct_entry,
                                         self.cep.migrating_specie).sites
            s0 = self.cep.base_struct_entry.structure.copy()
            s0.insert(0, 'Li', li_sites[0].frac_coords)
            for isite in li_sites[1:]:
                s1 = self.cep.base_struct_entry.structure.copy()
                s1.insert(0, 'Li', isite.frac_coords)
                self.assertTrue(self.rough_sm.fit(s0, s1))

    def test_get_full_sites(self):
        """
        For Mn6O5F7 there should be 8 symmetry inequivalent final relaxed positions
        """
        self.assertEqual(len(self.cep.full_sites), 8)

    def test_integration(self):
        """
        Sanity check: for a long enough diagonaly hop, if we turn the radius of the tube way up, it should cover the entire unit cell
        """
        self.cep._tube_radius = 10000
        total_chg_per_vol = self.cep.base_aeccar.data['total'].sum(
        ) / self.cep.base_aeccar.ngridpts / self.cep.base_aeccar.structure.volume
        self.assertAlmostEqual(
            self.cep._get_chg_between_sites_tube(self.cep.unique_hops[2]),
            total_chg_per_vol)

        self.cep._tube_radius = 2

        self.assertAlmostEqual(
            self.cep._get_chg_between_sites_tube(self.cep.unique_hops[0]),
            0.19531840655905952, 3)

    def test_populate_edges_with_chg_density_info(self):
        """
        Test that all of the sites with similar lengths have similar charge densities,
        this will not always be true, but it valid in this Mn6O5F7
        """
        self.cep.populate_edges_with_chg_density_info()
        length_vs_chg = list(
            sorted([(d['hop'].length, d['chg_total'])
                    for u, v, d in self.cep.s_graph.graph.edges(data=True)]))
        prv = None
        for len, chg in length_vs_chg:
            if prv is None:
                prv = (len, chg)
                continue

            if len / prv[0] < 1.05 and len / prv[0] > 0.95:
                self.assertAlmostEqual(chg, prv[1], 3)


if __name__ == '__main__':
    unittest.main()
