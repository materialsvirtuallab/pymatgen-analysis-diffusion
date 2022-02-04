# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.
from monty.serialization import loadfn
from pymatgen.entries.computed_entries import ComputedStructureEntry

from pymatgen.analysis.structure_matcher import StructureMatcher
import unittest
from pymatgen.core import Structure
import numpy as np
import os

from pymatgen.analysis.diffusion.utils.parse_entries import (
    get_inserted_on_base,
    process_entries,
    get_insertion_energy,
    get_sym_migration_ion_sites,
    _filter_and_merge,
)

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/test_files"

__author__ = "Jimmy Shen"
__version__ = "1.0"
__date__ = "April 10, 2019"


class ParseEntriesTest(unittest.TestCase):
    def setUp(self):
        d = loadfn(f"{dir_path}/parse_entry_test_vars.json")
        struct_uc = d["struct_uc"]
        self.li_ent = d["li_ent"]
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

        self.struct_inserted_1Li1 = get_inserted_on_base(self.base, self.inserted_1Li1, self.li_ent, self.sm)
        self.struct_inserted_1Li2 = get_inserted_on_base(self.base, self.inserted_1Li2, self.li_ent, self.sm)
        self.struct_inserted_2Li = get_inserted_on_base(self.base, self.inserted_2Li, self.li_ent, self.sm)

    def _is_valid_inserted_ent(self, mapped_struct):
        res = True
        for isite in mapped_struct.sites:
            if isite.species_string == "Li":
                tt = isite.frac_coords - np.floor(isite.frac_coords)
                if list(tt) in [[0.75, 0.75, 0.5], [0.25, 0.25, 0.5]]:
                    res = True
                else:
                    return False
        return res

    def test_get_inserted_on_base(self):
        mapped_struct = get_inserted_on_base(self.base, self.inserted_1Li1, self.li_ent, self.sm)
        self.assertTrue(self._is_valid_inserted_ent(mapped_struct))
        self.assertEqual(mapped_struct[0].properties["insertion_energy"], 5.0)
        mapped_struct = get_inserted_on_base(self.base, self.inserted_1Li2, self.li_ent, self.sm)
        self.assertTrue(self._is_valid_inserted_ent(mapped_struct))
        self.assertEqual(mapped_struct[0].properties["insertion_energy"], 7.0)
        mapped_struct = get_inserted_on_base(self.base, self.inserted_2Li, self.li_ent, self.sm)
        self.assertTrue(self._is_valid_inserted_ent(mapped_struct))
        self.assertEqual(mapped_struct[0].properties["insertion_energy"], 4.0)

    def test_process_ents(self):
        base_2_ent = ComputedStructureEntry(
            structure=self.base.structure * [[1, 1, 0], [1, -1, 0], [0, 0, 2]],
            energy=self.base.energy * 4,
        )
        res = process_entries(
            [base_2_ent, self.base],
            [self.inserted_2Li],
            migrating_ion_entry=self.li_ent,
        )
        for itr_group in res:
            for i_insert_site in itr_group["inserted"]:
                if i_insert_site.species_string == "Li":
                    self.assertEqual(i_insert_site.properties["insertion_energy"], 4)

    def test_filter_and_merge(self):
        combined_struct = Structure.from_sites(
            self.struct_inserted_1Li1.sites + self.struct_inserted_1Li2.sites + self.struct_inserted_2Li.sites
        )
        filtered_struct = _filter_and_merge(combined_struct)
        for i_insert_site in filtered_struct:
            if i_insert_site.species_string == "Li":
                self.assertIn(i_insert_site.properties["insertion_energy"], {4.5, 5.5})

    def test_get_insertion_energy(self):
        insert_energy = get_insertion_energy(self.base, self.inserted_1Li1, self.li_ent)
        basex2_ = ComputedStructureEntry(structure=self.base.structure * [1, 1, 2], energy=self.base.energy * 2)
        insert_energyx2 = get_insertion_energy(basex2_, self.inserted_1Li1, self.li_ent)
        self.assertAlmostEqual(insert_energyx2, insert_energy)
        self.assertAlmostEqual(insert_energy, 5)  # 3 + 2 where 2 is from the Li energy
        insert_energy = get_insertion_energy(self.base, self.inserted_2Li, self.li_ent)
        self.assertAlmostEqual(insert_energy, 4)

    def test_get_all_sym_sites(self):
        struct11 = get_sym_migration_ion_sites(self.base.structure, self.inserted_1Li1.structure, migrating_ion="Li")
        self.assertEqual(struct11.composition["Li"], 4)
        struct12 = get_sym_migration_ion_sites(self.base.structure, self.inserted_1Li2.structure, migrating_ion="Li")
        self.assertEqual(struct12.composition["Li"], 4)
        struct22 = get_sym_migration_ion_sites(self.base.structure, self.inserted_2Li.structure, migrating_ion="Li")
        self.assertEqual(struct22.composition["Li"], 8)


if __name__ == "__main__":
    unittest.main()
