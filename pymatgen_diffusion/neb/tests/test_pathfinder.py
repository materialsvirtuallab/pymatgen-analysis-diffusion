# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

from __future__ import division, unicode_literals

from pymatgen.core import Structure
from pymatgen_diffusion.neb.pathfinder import IDPPSolver, DistinctPathFinder
from pymatgen.util.testing import PymatgenTest
import unittest
import numpy as np
import os
import glob

__author__ = "Iek-Heng Chu"
__version__ = "1.0"
__date__ = "March 14, 2017"

import os
from pymatgen_diffusion.neb.pathfinder import combine_neb_plots

def get_path(path_str, dirname="./"):
    cwd = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(cwd, dirname, path_str)
    return path


class IDPPSolverTest(unittest.TestCase):
    init_struct = Structure.from_file(get_path("CONTCAR-0",
                                               dirname="pathfinder_files"))
    final_struct = Structure.from_file(get_path("CONTCAR-1",
                                                dirname="pathfinder_files"))

    def test_idpp_from_ep(self):
        obj = IDPPSolver.from_endpoints([self.init_struct, self.final_struct],
                                        nimages=3, sort_tol=1.0)
        new_path = obj.run(maxiter=5000, tol=1e-5, gtol=1e-3, step_size=0.05,
                           max_disp=0.05, spring_const=5.0, species=["Li"])

        self.assertEqual(len(new_path), 5)
        self.assertEqual(new_path[1].num_sites, 111)
        self.assertTrue(np.allclose(new_path[0][2].frac_coords,
                                    np.array([0.50000014, 0.99999998, 0.74999964])))
        self.assertTrue(np.allclose(new_path[1][0].frac_coords,
                                    np.array([0.482439, 0.68264727, 0.26525066])))
        self.assertTrue(np.allclose(new_path[2][10].frac_coords,
                                    np.array([0.50113915, 0.74958704, 0.75147021])))
        self.assertTrue(np.allclose(new_path[3][22].frac_coords,
                                    np.array([0.28422885, 0.62568764, 0.98975444])))
        self.assertTrue(np.allclose(new_path[4][47].frac_coords,
                                    np.array([0.59767531, 0.12640952, 0.37745006])))

        pass

    def test_idpp(self):
        images = self.init_struct.interpolate(self.final_struct, nimages=4,
                                              autosort_tol=1.0)
        obj = IDPPSolver(images)
        new_path = obj.run(maxiter=5000, tol=1e-5, gtol=1e-3, step_size=0.05,
                           max_disp=0.05, spring_const=5.0, species=["Li"])

        self.assertEqual(len(new_path), 5)
        self.assertEqual(new_path[1].num_sites, 111)
        self.assertTrue(np.allclose(new_path[0][2].frac_coords,
                                    np.array([0.50000014, 0.99999998, 0.74999964])))
        self.assertTrue(np.allclose(new_path[1][0].frac_coords,
                                    np.array([0.482439, 0.68264727, 0.26525066])))
        self.assertTrue(np.allclose(new_path[4][47].frac_coords,
                                    np.array([0.59767531, 0.12640952, 0.37745006])))


class DistinctPathFinderTest(PymatgenTest):
    def test_get_paths(self):
        s = self.get_structure("LiFePO4")
        # Only one path in LiFePO4 with 4 A.
        p = DistinctPathFinder(s, "Li", max_path_length=4)
        paths = p.get_paths()
        self.assertEqual(len(paths), 1)

        # Make sure this is robust to supercells.
        s.make_supercell((2, 2, 1))
        p = DistinctPathFinder(s, "Li", max_path_length=4)
        paths = p.get_paths()
        self.assertEqual(len(paths), 1)

        ss = paths[0].get_structures(vac_mode=False)
        self.assertEqual(len(ss), 7)

        paths[0].write_path("pathfindertest_noidpp_vac.cif", idpp=False)
        paths[0].write_path("pathfindertest_idpp_vac.cif", idpp=True)
        paths[0].write_path("pathfindertest_idpp_nonvac.cif", idpp=True, vac_mode=False)
        self.assertEqual(str(paths[0]),
                         "Path of 3.0328 A from Li [0.000, 0.500, 1.000] (ind: 0, Wyckoff: 16a) to Li [-0.000, 0.250, 1.000] (ind: 0, Wyckoff: 16a)")

        p = DistinctPathFinder(s, "Li", max_path_length=6)
        paths = p.get_paths()
        self.assertEqual(len(paths), 4)

        s = self.get_structure("Graphite")

        # Only one path in graphite with 2 A.
        p = DistinctPathFinder(s, "C0+", max_path_length=2)
        paths = p.get_paths()
        self.assertEqual(len(paths), 1)

        s = self.get_structure("Li3V2(PO4)3")
        p = DistinctPathFinder(s, "Li0+", max_path_length=4)
        paths = p.get_paths()

        self.assertEqual(len(paths), 4)
        p.write_all_paths("pathfindertest_LVPO.cif", nimages=10, idpp=True)

        for f in glob.glob("pathfindertest_*.cif"):
            os.remove(f)

    def test_max_path_length(self):
        s = Structure.from_file(get_path("LYPS.cif",
                                         dirname="pathfinder_files"))
        dp1 = DistinctPathFinder(s, "Li", perc_mode="1d")
        self.assertAlmostEqual(dp1.max_path_length, 4.11375354207, 7)
        dp2 = DistinctPathFinder(s, "Li", 5, perc_mode="1d")
        self.assertAlmostEqual(dp2.max_path_length, 5.0, 7)

def test_combine_neb_plots():
    test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            'test_files', 'neb')
    combine_neb_plots([test_dir, test_dir])

test_combine_neb_plots()

if __name__ == '__main__':
    unittest.main()
