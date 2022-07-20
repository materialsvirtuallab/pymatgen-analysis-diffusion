import os
import unittest
from monty.serialization import dumpfn, loadfn
from pymatgen.util.testing import PymatgenTest

module_dir = os.path.dirname(os.path.abspath(__file__))


class PathfinderTest(PymatgenTest):
    def test_mhop_msonable(self):
        spinel_mg = loadfn(module_dir + "/migration_graph_spinel_MgMn2O4.json")
        hop = spinel_mg.unique_hops[0]["hop"]
        hop_dict = hop.as_dict()

        assert type(hop_dict) == dict


if __name__ == "__main__":
    unittest.main()
