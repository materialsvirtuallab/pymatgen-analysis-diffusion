# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.
import os
import unittest

import numpy as np
from monty.serialization import loadfn
from pymatgen.core import Structure, PeriodicSite
from pymatgen.io.vasp import Chgcar

from pymatgen.analysis.diffusion.neb.full_path_mapper import (
    ChargeBarrierGraph,
    MigrationGraph,
    MigrationHop,
    get_hop_site_sequence,
)
from pymatgen.analysis.diffusion.neb.periodic_dijkstra import _get_adjacency_with_images

dir_path = os.path.dirname(os.path.realpath(__file__))

__author__ = "Jimmy Shen"
__version__ = "1.0"
__date__ = "April 10, 2019"


class MigrationGraphSimpleTest(unittest.TestCase):
    def setUp(self):
        struct = Structure.from_file(f"{dir_path}/full_path_files/MnO2_full_Li.vasp")
        self.fpm = MigrationGraph.with_distance(structure=struct, migrating_specie="Li", max_distance=4)

    def test_get_pos_and_migration_path(self):
        """
        Make sure that we can populate the graph with MigrationHop Objects
        """
        self.fpm._get_pos_and_migration_path(0, 1, 1)
        self.assertAlmostEqual(self.fpm.migration_graph.graph[0][1][1]["hop"].length, 3.571248, 4)

    def test_get_summary_dict(self):
        summary_dict = self.fpm.get_summary_dict()
        self.assertTrue("hop_label", summary_dict["hops"][0])
        self.assertTrue("hop_label", summary_dict["unique_hops"][0])


class MigrationGraphFromEntriesTest(unittest.TestCase):
    def setUp(self):
        self.test_ents_MOF = loadfn(f"{dir_path}/full_path_files/Mn6O5F7_cat_migration.json")
        self.aeccar_MOF = Chgcar.from_file(f"{dir_path}/full_path_files/AECCAR_Mn6O5F7.vasp")
        self.li_ent = loadfn(f"{dir_path}/full_path_files/li_ent.json")["li_ent"]

        self.full_struct = MigrationGraph.get_structure_from_entries(
            base_entries=[self.test_ents_MOF["ent_base"]],
            inserted_entries=self.test_ents_MOF["one_cation"],
            migrating_ion_entry=self.li_ent,
        )[0]

    def test_migration_graph_construction(self):
        self.assertEqual(self.full_struct.composition["Li"], 8)
        mg = MigrationGraph.with_distance(self.full_struct, migrating_specie="Li", max_distance=4.0)
        self.assertEqual(len(mg.migration_graph.structure), 8)


class MigrationGraphComplexTest(unittest.TestCase):
    def setUp(self):
        struct = Structure.from_file(f"{dir_path}/full_path_files/MnO2_full_Li.vasp")
        self.fpm_li = MigrationGraph.with_distance(structure=struct, migrating_specie="Li", max_distance=4)

        # Particularity difficult pathfinding since both the starting and ending positions are outside the unit cell
        struct = Structure.from_file(f"{dir_path}/full_path_files/Mg_2atom.vasp")
        self.fpm_mg = MigrationGraph.with_distance(structure=struct, migrating_specie="Mg", max_distance=2)

    def test_group_and_label_hops(self):
        """
        Check that the set of end points in a group of similiarly labeled hops are all the same
        """
        edge_labs = np.array([d["hop_label"] for u, v, d in self.fpm_li.migration_graph.graph.edges(data=True)])

        site_labs = np.array(
            [
                (
                    d["hop"].symm_structure.wyckoff_symbols[d["hop"].iindex],
                    d["hop"].symm_structure.wyckoff_symbols[d["hop"].eindex],
                )
                for u, v, d in self.fpm_li.migration_graph.graph.edges(data=True)
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
        unique_list = [v for k, v in self.fpm_li.unique_hops.items()]
        all_pairs = [(mg1, mg2) for i1, mg1 in enumerate(unique_list) for mg2 in unique_list[i1 + 1 :]]

        for migration_path in all_pairs:
            self.assertNotEqual(migration_path[0]["hop"], migration_path[1]["hop"])

    def test_add_data_to_similar_edges(self):
        # passing normal data
        self.fpm_li.add_data_to_similar_edges(0, {"key0": "data"})
        for u, v, d in self.fpm_li.migration_graph.graph.edges(data=True):
            if d["hop_label"] == 0:
                self.assertEqual(d["key0"], "data")

        # passing ordered list data
        migration_path = self.fpm_li.unique_hops[1]["hop"]
        self.fpm_li.add_data_to_similar_edges(1, {"key1": [1, 2, 3]}, m_path=migration_path)
        for u, v, d in self.fpm_li.migration_graph.graph.edges(data=True):
            if d["hop_label"] == 1:
                self.assertEqual(d["key1"], [1, 2, 3])

        # passing ordered list with direction
        migration_path_reversed = MigrationHop(
            isite=migration_path.esite,
            esite=migration_path.isite,
            symm_structure=migration_path.symm_structure,
        )
        self.fpm_li.add_data_to_similar_edges(2, {"key2": [1, 2, 3]}, m_path=migration_path_reversed)
        for u, v, d in self.fpm_li.migration_graph.graph.edges(data=True):
            if d["hop_label"] == 2:
                self.assertEqual(d["key2"], [3, 2, 1])

    def test_assign_cost_to_graph(self):
        self.fpm_li.assign_cost_to_graph()  # use 'hop_distance'
        for u, v, d in self.fpm_li.migration_graph.graph.edges(data=True):
            self.assertAlmostEqual(d["cost"], d["hop_distance"], 4)

        self.fpm_li.assign_cost_to_graph(cost_keys=["hop_distance", "hop_distance"])
        for u, v, d in self.fpm_li.migration_graph.graph.edges(data=True):
            self.assertAlmostEqual(d["cost"], d["hop_distance"] ** 2, 4)

    def test_periodic_dijkstra(self):
        self.fpm_li.assign_cost_to_graph()  # use 'hop_distance'

        # test the connection graph
        sgraph = self.fpm_li.migration_graph
        G = sgraph.graph.to_undirected()
        conn_dict = _get_adjacency_with_images(G)
        for u in conn_dict.keys():
            for v in conn_dict[u]:
                for k, d in conn_dict[u][v].items():
                    neg_image = tuple(-dim_ for dim_ in d["to_jimage"])
                    opposite_connections = [d2_["to_jimage"] for k2_, d2_ in conn_dict[v][u].items()]
                    self.assertIn(neg_image, opposite_connections)

    def test_get_path(self):
        self.fpm_li.assign_cost_to_graph()  # use 'hop_distance'
        paths = [*self.fpm_li.get_path()]
        p_strings = {"->".join(map(str, get_hop_site_sequence(ipath, start_u=u))) for u, ipath in paths}
        self.assertIn("5->7->5", p_strings)
        # convert each pathway to a string representation
        paths = [*self.fpm_li.get_path(max_val=2.0)]
        p_strings = {"->".join(map(str, get_hop_site_sequence(ipath, start_u=u))) for u, ipath in paths}

        # After checking trimming the graph more hops are needed for the same path
        self.assertIn("5->3->7->2->5", p_strings)

        self.fpm_mg.assign_cost_to_graph()  # use 'hop_distance'
        paths = [*self.fpm_mg.get_path()]
        p_strings = {"->".join(map(str, get_hop_site_sequence(ipath, start_u=u))) for u, ipath in paths}
        self.assertIn("1->0->1", p_strings)

    def test_not_matching_first(self):
        structure = Structure.from_file(f"{dir_path}/pathfinder_files/Li6MnO4.json")
        fpm_lmo = MigrationGraph.with_distance(structure, "Li", max_distance=4)
        for u, v, d in fpm_lmo.migration_graph.graph.edges(data=True):
            self.assertIn(d["hop"].eindex, {0, 1})


class ChargeBarrierGraphTest(unittest.TestCase):
    def setUp(self):
        self.full_sites_MOF = loadfn(f"{dir_path}/full_path_files/LixMn6O5F7_full_sites.json")
        self.aeccar_MOF = Chgcar.from_file(f"{dir_path}/full_path_files/AECCAR_Mn6O5F7.vasp")
        self.cbg = ChargeBarrierGraph.with_distance(
            structure=self.full_sites_MOF,
            migrating_specie="Li",
            max_distance=4,
            potential_field=self.aeccar_MOF,
            potential_data_key="total",
        )
        self.cbg._tube_radius = 10000

    def test_integration(self):
        """
        Sanity check: for a long enough diagonaly hop, if we turn the radius of the tube way up, it should cover the entire unit cell
        """
        total_chg_per_vol = (
            self.cbg.potential_field.data["total"].sum()
            / self.cbg.potential_field.ngridpts
            / self.cbg.potential_field.structure.volume
        )
        self.assertAlmostEqual(
            self.cbg._get_chg_between_sites_tube(self.cbg.unique_hops[2]["hop"]),
            total_chg_per_vol,
        )

        self.cbg._tube_radius = 2
        # self.cbg.populate_edges_with_chg_density_info()

        # find this particular hop
        ipos = [0.33079153, 0.18064031, 0.67945924]
        epos = [0.33587514, -0.3461259, 1.15269302]
        isite = PeriodicSite("Li", ipos, self.cbg.structure.lattice)
        esite = PeriodicSite("Li", epos, self.cbg.structure.lattice)
        ref_hop = MigrationHop(isite=isite, esite=esite, symm_structure=self.cbg.symm_structure)
        hop_idx = -1
        for k, d in self.cbg.unique_hops.items():
            if d["hop"] == ref_hop:
                hop_idx = k

        self.assertAlmostEqual(
            self.cbg._get_chg_between_sites_tube(self.cbg.unique_hops[hop_idx]["hop"]),
            0.188952739835188,
            3,
        )

    def test_populate_edges_with_chg_density_info(self):
        """
        Test that all of the sites with similar lengths have similar charge densities,
        this will not always be true, but it valid in this Mn6O5F7
        """
        self.cbg.populate_edges_with_chg_density_info()
        length_vs_chg = list(
            sorted([(d["hop"].length, d["chg_total"]) for u, v, d in self.cbg.migration_graph.graph.edges(data=True)])
        )
        prv = None
        for length, chg in length_vs_chg:
            if prv is None:
                prv = (length, chg)
                continue

            if 1.05 > length / prv[0] > 0.95:
                self.assertAlmostEqual(chg, prv[1], 3)

    def test_get_summary_dict(self):
        summary_dict = self.cbg.get_summary_dict()
        self.assertTrue("chg_total", summary_dict["hops"][0])
        self.assertTrue("chg_total", summary_dict["unique_hops"][0])


if __name__ == "__main__":
    unittest.main()
