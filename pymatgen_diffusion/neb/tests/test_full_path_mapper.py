# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

from pymatgen_diffusion.neb.periodic_dijkstra import _get_adjacency_with_images
from pymatgen_diffusion.neb.full_path_mapper import (
    FullPathMapper,
    ComputedEntryPath,
    get_all_sym_sites,
    get_hop_site_sequence,
    MigrationPath,
)

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
        base_struct = Structure.from_file(f"{dir_path}/full_path_files/MnO2_base.vasp")
        sites_struct = Structure.from_file(
            f"{dir_path}/full_path_files/MnO2_only_Li.vasp"
        )
        self.fpm = FullPathMapper(
            base_structure=base_struct,
            mobile_specie="Li",
            sites_structure=sites_struct,
            max_hop_length=4,
        )

    def test_get_pos_and_migration_path(self):
        """
        Make sure that we can populate the graph with MigrationPath Objects
        """
        self.fpm._get_pos_and_migration_path(0, 1, 1)
        self.assertAlmostEqual(
            self.fpm.s_graph.graph[0][1][1]["hop"].length, 3.571248, 4
        )


class FullPathMapperComplexTest(unittest.TestCase):
    def setUp(self):
        base_struct = Structure.from_file(f"{dir_path}/full_path_files/MnO2_base.vasp")
        sites_struct = Structure.from_file(
            f"{dir_path}/full_path_files/MnO2_only_Li.vasp"
        )
        self.fpm_li = FullPathMapper(
            base_structure=base_struct,
            mobile_specie="Li",
            sites_structure=sites_struct,
            max_hop_length=4,
        )
        self.fpm_li.populate_edges_with_migration_paths()
        self.fpm_li.group_and_label_hops()
        self.fpm_li._populate_unique_hops_dict()

        # Particularity difficult pathfinding since both the starting and ending positions are outside the unit cell
        mg_struct = Structure.from_file(f"{dir_path}/full_path_files/Mg_2atom.vasp")
        au_struct = Structure.from_file(
            f"{dir_path}/full_path_files/Mg_2atom_base.vasp"
        )
        self.fpm_mg = FullPathMapper(
            base_structure=au_struct,
            mobile_specie="Mg",
            sites_structure=mg_struct,
            max_hop_length=2,
        )
        self.fpm_mg.populate_edges_with_migration_paths()
        self.fpm_mg.group_and_label_hops()
        self.fpm_mg._populate_unique_hops_dict()

    def test_group_and_label_hops(self):
        """
        Check that the set of end points in a group of similiarly labeled hops are all the same
        """
        edge_labs = np.array(
            [d["hop_label"] for u, v, d in self.fpm_li.s_graph.graph.edges(data=True)]
        )

        site_labs = np.array(
            [
                (
                    d["hop"].symm_structure.wyckoff_symbols[d["hop"].iindex],
                    d["hop"].symm_structure.wyckoff_symbols[d["hop"].eindex],
                )
                for u, v, d in self.fpm_li.s_graph.graph.edges(data=True)
            ]
        )

        for itr in range(edge_labs.max()):
            sub_set = site_labs[edge_labs == itr]
            for end_point_labels in sub_set:
                self.assertTrue(sorted(end_point_labels) == sorted(sub_set[0]))

    def test_unique_hops_dict(self):
        """
        Check that the unique hops are inequilvalent
        """
        self.fpm_li._populate_unique_hops_dict()
        unique_list = [v for k, v in self.fpm_li.unique_hops.items()]
        all_pairs = [
            (mg1, mg2)
            for i1, mg1 in enumerate(unique_list)
            for mg2 in unique_list[i1 + 1 :]
        ]

        for migration_path in all_pairs:
            self.assertNotEqual(migration_path[0]["hop"], migration_path[1]["hop"])

    def test_add_data_to_similar_edges(self):
        # passing normal data
        self.fpm_li.add_data_to_similar_edges(0, {"key0": "data"})
        for u, v, d in self.fpm_li.s_graph.graph.edges(data=True):
            if d["hop_label"] == 0:
                self.assertEqual(d["key0"], "data")

        # passing ordered list data
        migration_path = self.fpm_li.unique_hops[1]["hop"]
        self.fpm_li.add_data_to_similar_edges(
            1, {"key1": [1, 2, 3]}, m_path=migration_path
        )
        for u, v, d in self.fpm_li.s_graph.graph.edges(data=True):
            if d["hop_label"] == 1:
                self.assertEqual(d["key1"], [1, 2, 3])

        # passing ordered list with direction
        migration_path_reversed = MigrationPath(
            isite=migration_path.esite,
            esite=migration_path.isite,
            symm_structure=migration_path.symm_structure,
        )
        self.fpm_li.add_data_to_similar_edges(
            2, {"key2": [1, 2, 3]}, m_path=migration_path_reversed
        )
        for u, v, d in self.fpm_li.s_graph.graph.edges(data=True):
            if d["hop_label"] == 2:
                self.assertEqual(d["key2"], [3, 2, 1])

    def test_assign_cost_to_graph(self):
        self.fpm_li.assign_cost_to_graph()  # use 'hop_distance'
        for u, v, d in self.fpm_li.s_graph.graph.edges(data=True):
            self.assertAlmostEqual(d["cost"], d["properties"]["hop_distance"], 4)

        self.fpm_li.assign_cost_to_graph(cost_keys=["hop_distance", "hop_distance"])
        for u, v, d in self.fpm_li.s_graph.graph.edges(data=True):
            self.assertAlmostEqual(d["cost"], d["properties"]["hop_distance"] ** 2, 4)

    def test_periodic_dijkstra(self):
        self.fpm_li.assign_cost_to_graph()  # use 'hop_distance'

        # test the connection graph
        sgraph = self.fpm_li.s_graph
        G = sgraph.graph.to_undirected()
        conn_dict = _get_adjacency_with_images(G)
        for u in conn_dict.keys():
            for v in conn_dict[u]:
                for k, d in conn_dict[u][v].items():
                    neg_image = tuple(-dim_ for dim_ in d["to_jimage"])
                    opposite_connections = [
                        d2_["to_jimage"] for k2_, d2_ in conn_dict[v][u].items()
                    ]
                    self.assertIn(neg_image, opposite_connections)

    def test_get_path(self):
        self.fpm_li.assign_cost_to_graph()  # use 'hop_distance'
        paths = [*self.fpm_li.get_path()]
        p_strings = {
            "->".join(map(str, get_hop_site_sequence(ipath, start_u=u)))
            for u, ipath in paths
        }
        self.assertIn("5->7->5", p_strings)
        # convert each pathway to a string representation
        paths = [*self.fpm_li.get_path(max_val=2.0)]
        p_strings = {
            "->".join(map(str, get_hop_site_sequence(ipath, start_u=u)))
            for u, ipath in paths
        }

        # After checking trimming the graph more hops are needed for the same path
        self.assertIn("5->3->7->2->5", p_strings)

        self.fpm_mg.assign_cost_to_graph()  # use 'hop_distance'
        paths = [*self.fpm_mg.get_path()]
        p_strings = {
            "->".join(map(str, get_hop_site_sequence(ipath, start_u=u)))
            for u, ipath in paths
        }
        self.assertIn("1->0->1", p_strings)


class ComputedEntryPathTest(unittest.TestCase):
    def setUp(self):
        self.test_ents_MOF = loadfn(
            f"{dir_path}/full_path_files/Mn6O5F7_cat_migration.json"
        )
        self.aeccar_MOF = Chgcar.from_file(
            f"{dir_path}/full_path_files/AECCAR_Mn6O5F7.vasp"
        )
        self.cep = ComputedEntryPath(
            base_struct_entry=self.test_ents_MOF["ent_base"],
            mobile_specie="Li",
            single_cat_entries=self.test_ents_MOF["one_cation"],
            base_aeccar=self.aeccar_MOF,
        )

        # structure matcher used for validation
        self.rough_sm = StructureMatcher(
            comparator=ElementComparator(),
            primitive_cell=False,
            ltol=0.5,
            stol=0.5,
            angle_tol=7,
        )

    def test_get_all_sym_sites(self):
        """
        Inserting a Li into each one of the proposed structures should result in an equivalent structure
        (using sloppier tolerences)
        """
        for ent in self.cep.translated_single_cat_entries:
            li_sites = get_all_sym_sites(
                ent, self.cep.base_struct_entry, self.cep.mobile_specie
            ).sites
            s0 = self.cep.base_struct_entry.structure.copy()
            s0.insert(0, "Li", li_sites[0].frac_coords)
            for isite in li_sites[1:]:
                s1 = self.cep.base_struct_entry.structure.copy()
                s1.insert(0, "Li", isite.frac_coords)
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
        total_chg_per_vol = (
            self.cep.base_aeccar.data["total"].sum()
            / self.cep.base_aeccar.ngridpts
            / self.cep.base_aeccar.structure.volume
        )
        self.assertAlmostEqual(
            self.cep._get_chg_between_sites_tube(self.cep.unique_hops[2]["hop"]),
            total_chg_per_vol,
        )

        self.cep._tube_radius = 2

        self.assertAlmostEqual(
            self.cep._get_chg_between_sites_tube(self.cep.unique_hops[0]["hop"]),
            0.188952739835188,
            3,
        )

    def test_populate_edges_with_chg_density_info(self):
        """
        Test that all of the sites with similar lengths have similar charge densities,
        this will not always be true, but it valid in this Mn6O5F7
        """
        self.cep.populate_edges_with_chg_density_info()
        length_vs_chg = list(
            sorted(
                [
                    (d["hop"].length, d["chg_total"])
                    for u, v, d in self.cep.s_graph.graph.edges(data=True)
                ]
            )
        )
        prv = None
        for len, chg in length_vs_chg:
            if prv is None:
                prv = (len, chg)
                continue

            if len / prv[0] < 1.05 and len / prv[0] > 0.95:
                self.assertAlmostEqual(chg, prv[1], 3)


if __name__ == "__main__":
    unittest.main()
