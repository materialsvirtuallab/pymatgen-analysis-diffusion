# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.
from __future__ import annotations

import glob
import os
import unittest

import numpy as np

from pymatgen.analysis.diffusion.neb.full_path_mapper import MigrationGraph
from pymatgen.analysis.diffusion.neb.pathfinder import DistinctPathFinder, IDPPSolver, MigrationHop
from pymatgen.analysis.diffusion.utils.supercells import get_start_end_structures
from pymatgen.core import Composition, PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.testing import PymatgenTest

__author__ = "Iek-Heng Chu"
__version__ = "1.0"
__date__ = "March 14, 2017"


def get_path(path_str, dirname="./"):
    cwd = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(cwd, dirname, path_str)


class IDPPSolverTest(unittest.TestCase):
    init_struct = Structure.from_file(get_path("CONTCAR-0", dirname="pathfinder_files"))
    final_struct = Structure.from_file(get_path("CONTCAR-1", dirname="pathfinder_files"))

    def test_idpp_from_ep(self):
        obj = IDPPSolver.from_endpoints([self.init_struct, self.final_struct], nimages=3, sort_tol=1.0)
        new_path = obj.run(
            maxiter=5000,
            tol=1e-5,
            gtol=1e-3,
            step_size=0.05,
            max_disp=0.05,
            spring_const=5.0,
            species=["Li"],
        )

        assert len(new_path) == 5
        assert new_path[1].num_sites == 111
        assert np.allclose(new_path[0][2].frac_coords, np.array([0.50000014, 0.99999998, 0.74999964]))
        assert np.allclose(new_path[1][0].frac_coords, np.array([0.482439, 0.68264727, 0.26525066]))
        assert np.allclose(new_path[2][10].frac_coords, np.array([0.50113915, 0.74958704, 0.75147021]))
        assert np.allclose(new_path[3][22].frac_coords, np.array([0.28422885, 0.62568764, 0.98975444]))
        assert np.allclose(new_path[4][47].frac_coords, np.array([0.59767531, 0.12640952, 0.37745006]))

    def test_idpp_from_ep_diff_latt(self):
        # This is the same test as test_idpp_from_ep where we want to interpolate
        # different lattices.

        # make a copy of the final_struct and distort the lattice by 5%
        final_struct_strain = self.final_struct.copy()
        final_struct_strain.apply_strain(0.05)

        obj = IDPPSolver.from_endpoints(
            [self.init_struct, final_struct_strain],
            nimages=3,
            sort_tol=1.0,
            interpolate_lattices=True,
        )
        new_path = obj.run(
            maxiter=5000,
            tol=1e-5,
            gtol=1e-3,
            step_size=0.05,
            max_disp=0.05,
            spring_const=5.0,
            species=["Li"],
        )

        assert len(new_path) == 5
        assert new_path[1].num_sites == 111

        # For checking coords, the endpoints will have the same frac coords as the
        # original test, whereas the midpoints will not have altered coord because
        # of the differing lattice.
        assert np.allclose(new_path[0][2].frac_coords, np.array([0.50000014, 0.99999998, 0.74999964]))
        assert np.allclose(new_path[1][0].frac_coords, np.array([0.47483465, 0.6740367, 0.26270862]))
        assert np.allclose(new_path[2][10].frac_coords, np.array([0.48839604, 0.73135841, 0.73315875]))
        assert np.allclose(new_path[3][22].frac_coords, np.array([0.27395552, 0.60307242, 0.95398018]))
        assert np.allclose(new_path[4][47].frac_coords, np.array([0.59767531, 0.12640952, 0.37745006]))

        # ensure the lattices are interpolated
        assert np.allclose(
            new_path[0].lattice.matrix,
            np.array(
                [
                    [10.4117233, 0.0, 0.0],
                    [0.0, 12.07835177, 0.0],
                    [0.0, 0.0, 9.47759962],
                ]
            ),
        )
        assert np.allclose(
            new_path[1].lattice.matrix,
            np.array(
                [
                    [10.54186984, 0.0, 0.0],
                    [0.0, 12.22933117, 0.0],
                    [0.0, 0.0, 9.59606961],
                ]
            ),
        )
        assert np.allclose(
            new_path[2].lattice.matrix,
            np.array(
                [
                    [10.67201638, 0.0, 0.0],
                    [0.0, 12.38031057, 0.0],
                    [0.0, 0.0, 9.71453961],
                ]
            ),
        )
        assert np.allclose(
            new_path[3].lattice.matrix,
            np.array(
                [
                    [10.80216292, 0.0, 0.0],
                    [0.0, 12.53128996, 0.0],
                    [0.0, 0.0, 9.8330096],
                ]
            ),
        )
        assert np.allclose(
            new_path[4].lattice.matrix,
            np.array(
                [
                    [10.93230946, 0.0, 0.0],
                    [0.0, 12.68226936, 0.0],
                    [0.0, 0.0, 9.9514796],
                ]
            ),
        )

    def test_idpp(self):
        images = self.init_struct.interpolate(self.final_struct, nimages=4, autosort_tol=1.0)
        obj = IDPPSolver(images)
        new_path = obj.run(
            maxiter=5000,
            tol=1e-5,
            gtol=1e-3,
            step_size=0.05,
            max_disp=0.05,
            spring_const=5.0,
            species=["Li"],
        )

        assert len(new_path) == 5
        assert new_path[1].num_sites == 111
        assert np.allclose(new_path[0][2].frac_coords, np.array([0.50000014, 0.99999998, 0.74999964]))
        assert np.allclose(new_path[1][0].frac_coords, np.array([0.482439, 0.68264727, 0.26525066]))
        assert np.allclose(new_path[4][47].frac_coords, np.array([0.59767531, 0.12640952, 0.37745006]))


class DistinctPathFinderTest(PymatgenTest):
    def test_get_paths(self):
        s = self.get_structure("LiFePO4")
        # Only one path in LiFePO4 with 4 A.
        p = DistinctPathFinder(s, "Li", max_path_length=4)
        paths = p.get_paths()
        assert len(paths) == 1

        # Make sure this is robust to supercells.
        s.make_supercell((2, 2, 1))
        p = DistinctPathFinder(s, "Li", max_path_length=4)
        paths = p.get_paths()
        assert len(paths) == 1

        ss = paths[0].get_structures(vac_mode=False)
        assert len(ss) == 7

        paths[0].write_path("pathfindertest_noidpp_vac.cif", idpp=False)
        paths[0].write_path("pathfindertest_idpp_vac.cif", idpp=True)
        paths[0].write_path("pathfindertest_idpp_nonvac.cif", idpp=True, vac_mode=False)
        assert (
            str(paths[0])
            == "Path of 3.0329 A from Li [0.000, 0.500, 1.000] (ind: 0, Wyckoff: 16a) to Li [-0.000, 0.750, 1.000] (ind: 0, Wyckoff: 16a)"
        )

        p = DistinctPathFinder(s, "Li", max_path_length=6)
        paths = p.get_paths()
        assert len(paths) == 4

        s = self.get_structure("Graphite")

        # Only one path in graphite with 2 A.
        p = DistinctPathFinder(s, "C0+", max_path_length=2)
        paths = p.get_paths()
        assert len(paths) == 1

        s = self.get_structure("Li3V2(PO4)3")
        p = DistinctPathFinder(s, "Li0+", max_path_length=4)
        paths = p.get_paths()

        assert len(paths) == 4
        p.write_all_paths("pathfindertest_LVPO.cif", nimages=10, idpp=True)

        for f in glob.glob("pathfindertest_*.cif"):
            os.remove(f)

    def test_max_path_length(self):
        s = Structure.from_file(get_path("LYPS.cif", dirname="pathfinder_files"))
        dp1 = DistinctPathFinder(s, "Li", perc_mode="1d")
        self.assertAlmostEqual(dp1.max_path_length, 4.11375354207, 7)
        dp2 = DistinctPathFinder(s, "Li", 5, perc_mode="1d")
        self.assertAlmostEqual(dp2.max_path_length, 5.0, 7)


class MigrationHopTest(PymatgenTest):
    def setUp(self):
        self.lifepo = self.get_structure("LiFePO4")
        m_graph = MigrationGraph.with_distance(self.lifepo, max_distance=4.0, migrating_specie="Li")
        gen = iter(m_graph.m_graph.graph.edges(data=True))
        u, v, d = next(gen)
        i_site = PeriodicSite("Li", coords=d["ipos"], lattice=self.lifepo.lattice)
        e_site = PeriodicSite("Li", coords=d["epos"], lattice=self.lifepo.lattice)
        a = SpacegroupAnalyzer(self.lifepo)
        symm_structure = a.get_symmetrized_structure()
        self.m_hop = MigrationHop(i_site, e_site, symm_structure)

    def test_msonable(self):
        hop_dict = self.m_hop.as_dict()
        assert isinstance(hop_dict, dict)

    def test_get_start_end_structs_from_hop(self):
        dist_ref = self.m_hop.length
        base = self.lifepo.copy()
        base.remove_species(["Li"])
        start, end, b_sc = get_start_end_structures(
            self.m_hop.isite,
            self.m_hop.esite,
            base_struct=base,
            sc_mat=[[2, 1, 0], [-1, 1, 0], [0, 0, 1]],
            vac_mode=False,
        )
        start_site = next(filter(lambda x: x.species_string == "Li", start.sites))
        end_site = next(filter(lambda x: x.species_string == "Li", end.sites))
        self.assertAlmostEqual(start_site.distance(end_site), dist_ref, 3)

    def test_get_start_end_structs_from_hop_vac(self):
        dist_ref = self.m_hop.length
        start, end, b_sc = get_start_end_structures(
            self.m_hop.isite,
            self.m_hop.esite,
            base_struct=self.lifepo,
            sc_mat=[[2, 1, 0], [-1, 1, 0], [0, 0, 2]],
            vac_mode=True,
        )
        s1 = set()
        s2 = set()
        for itr_start, start_site in enumerate(start.sites):
            for itr_end, end_site in enumerate(end.sites):
                dist_ = start_site.distance(end_site)
                if dist_ < 1e-5:
                    s1.add(itr_start)
                    s2.add(itr_end)
        s1 = sorted(s1, reverse=True)
        s2 = sorted(s2, reverse=True)
        start.remove_sites(s1)
        end.remove_sites(s2)
        self.assertAlmostEqual(start.sites[0].distance(end.sites[0]), dist_ref, 3)

    def test_get_sc_structures(self):
        dist_ref = self.m_hop.length
        start, end, b_sc = self.m_hop.get_sc_structures(vac_mode=False)
        start_site = next(filter(lambda x: x.species_string == "Li", start.sites))
        end_site = next(filter(lambda x: x.species_string == "Li", end.sites))
        assert start.composition == end.composition == Composition("Li1 Fe24 P24 O96")
        assert b_sc.composition == Composition("Fe24 P24 O96")
        self.assertAlmostEqual(start_site.distance(end_site), dist_ref, 3)

    def test_get_sc_structures_vacmode(self):
        start, end, b_sc = self.m_hop.get_sc_structures(vac_mode=True)
        assert start.composition == end.composition == Composition("Li23 Fe24 P24 O96")
        assert b_sc.composition == Composition("Li24 Fe24 P24 O96")
