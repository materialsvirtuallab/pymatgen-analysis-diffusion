from __future__ import annotations

import os

from monty.serialization import loadfn

from pymatgen.util.testing import PymatgenTest

module_dir = os.path.dirname(os.path.abspath(__file__))


class PathfinderTest(PymatgenTest):
    def test_mhop_msonable(self) -> None:
        file_path = os.path.join(module_dir, "migration_graph_spinel_MgMn2O4.json")
        spinel_mg = loadfn(file_path)
        hop = spinel_mg.unique_hops[0]["hop"]
        hop_dict = hop.as_dict()

        assert isinstance(hop_dict, dict)
