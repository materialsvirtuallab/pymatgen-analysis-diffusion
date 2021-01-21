# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.
from pymatgen.entries.computed_entries import ComputedStructureEntry

from pymatgen.analysis.structure_matcher import StructureMatcher
import unittest
from pymatgen import Structure
import numpy as np
import os

from pymatgen_diffusion.utils.parse_entries import get_inserted_on_base, process_ents

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../../neb/tests/"

__author__ = "Jimmy Shen"
__version__ = "1.0"
__date__ = "April 10, 2019"


class ParseEntriesSimplTest(unittest.TestCase):
    def setUp(self):
        struct_uc = Structure.from_dict(
            {
                "@module": "pymatgen.core.structure",
                "@class": "Structure",
                "charge": None,
                "lattice": {
                    "matrix": [[3, 0, 0], [1.5, 2.598, 0], [0, 0, 6]],
                    "a": 3,
                    "b": 3,
                    "c": 6,
                    "alpha": 90,
                    "beta": 90,
                    "gamma": 60,
                    "volume": 46.77,
                },
                "sites": [
                    {
                        "species": [{"element": "B", "occu": 1}],
                        "abc": [0, 0, 0.5],
                        "xyz": [0, 0, 3],
                        "label": "B",
                        "properties": {},
                    },
                    {
                        "species": [{"element": "N", "occu": 1}],
                        "abc": [0.5, 0.5, 0.5],
                        "xyz": [2.25, 1.299, 3],
                        "label": "N",
                        "properties": {},
                    },
                ],
            }
        )

        e_uc = 100
        self.base = ComputedStructureEntry(structure=struct_uc, energy=e_uc)

        sc = struct_uc * [2, 2, 2]
        sc.insert(0, "Li", [0.125, 0.125, 0.25])
        self.inserted_1Li1 = ComputedStructureEntry(structure=sc, energy=e_uc * 8 + 3)

        sc = struct_uc * [2, 2, 2]
        sc.insert(0, "Li", [0.375, 0.375, 0.25])
        self.inserted_1Li2 = ComputedStructureEntry(structure=sc, energy=e_uc * 8 + 5)

        sc = struct_uc * [2, 2, 2]
        sc.insert(0, "Li", [0.125, 0.125, 0.25])
        sc.insert(0, "Li", [0.375, 0.375, 0.25])
        self.inserted_2Li = ComputedStructureEntry(structure=sc, energy=e_uc * 8 + 4)

        self.sm = StructureMatcher(ignored_species=["Li"], primitive_cell=False)

    def _is_valid_inserted_ent(self, mapped_ents):
        res = True
        for ient in mapped_ents:
            for isite in ient.structure.sites:
                if isite.species_string == "Li":
                    tt = isite.frac_coords - np.floor(isite.frac_coords)
                    if list(tt) in [[0.75, 0.75, 0.5], [0.25, 0.25, 0.5]]:
                        res = True
                    else:
                        return False
        return res

    def test_get_inserted_on_base(self):
        mapped_ents = get_inserted_on_base(self.base, self.inserted_1Li1, self.sm)
        self.assertTrue(self._is_valid_inserted_ent(mapped_ents))
        self.assertEqual(mapped_ents[0].energy, 103)
        mapped_ents = get_inserted_on_base(self.base, self.inserted_1Li2, self.sm)
        self.assertTrue(self._is_valid_inserted_ent(mapped_ents))
        self.assertEqual(mapped_ents[0].energy, 105)
        mapped_ents = get_inserted_on_base(self.base, self.inserted_2Li, self.sm)
        self.assertTrue(self._is_valid_inserted_ent(mapped_ents))
        self.assertEqual(mapped_ents[0].energy, 102)

    def test_process_ents(self):
        base_2_ent = ComputedStructureEntry(
            structure=self.base.structure * [[1, 1, 0], [1, -1, 0], [0, 0, 2]],
            energy=self.base.energy * 4,
        )
        res = process_ents(
            [base_2_ent, self.base], [self.inserted_1Li1, self.inserted_2Li]
        )
        for itr_group in res:
            base_energy = itr_group["base"].energy
            for i_insert in itr_group["inserted"]:
                self.assertTrue(i_insert.energy - base_energy in [2, 3])


if __name__ == "__main__":
    unittest.main()
