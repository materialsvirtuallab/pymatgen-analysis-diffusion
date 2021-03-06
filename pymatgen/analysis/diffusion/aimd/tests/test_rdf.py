import os
import unittest

from monty.serialization import loadfn
import numpy as np
from pymatgen.core import Lattice, Structure

from pymatgen.analysis.diffusion.aimd.rdf import RadialDistributionFunctionFast


tests_dir = os.path.dirname(os.path.abspath(__file__))


class RDFTest(unittest.TestCase):
    def test_rdf(self):
        # Parse the DiffusionAnalyzer object from json file directly
        obj = loadfn(os.path.join(tests_dir, "cNa3PS4_pda.json"))

        structure_list = []
        for i, s in enumerate(obj.get_drift_corrected_structures()):
            structure_list.append(s)
            if i == 9:
                break
        species = ["Na", "P", "S"]

        # Test from_species
        obj = RadialDistributionFunctionFast(structures=structure_list, ngrid=101, rmax=10.0, sigma=0.1)

        r, s_na_rdf = obj.get_rdf("S", "Na")
        self.assertTrue(s_na_rdf.shape == (101,))

        self.assertAlmostEqual(r[np.argmax(s_na_rdf)], 2.9000, 4)

    def test_rdf_coordination_number(self):
        # create a simple cubic lattice
        coords = np.array([[0.5, 0.5, 0.5]])
        atom_list = ["S"]
        lattice = Lattice.from_parameters(a=1.0, b=1.0, c=1.0, alpha=90, beta=90, gamma=90)
        structure = Structure(lattice, atom_list, coords)
        rdf = RadialDistributionFunctionFast(structures=[structure], rmax=5.0, sigma=0.01, ngrid=500)
        self.assertEqual(np.round(rdf.get_coordination_number("S", "S")[1][110], 2), 6.0)


if __name__ == "__main__":
    unittest.main()
