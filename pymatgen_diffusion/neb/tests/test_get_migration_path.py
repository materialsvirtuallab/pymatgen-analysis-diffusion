# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.
import pytest

import os

from pymatgen import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen_diffusion.neb.get_migration_path import (
    get_cep_from_grouped_entries,
    get_sc_fromstruct,
    _get_sc_from_struct_ase,
)
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

__author__ = "Jimmy Shen"
__date__ = "Jan 28, 2021"


@pytest.fixture
def entries():
    struct = Structure.from_file(f"{dir_path}/full_path_files/MnO2_full_Li.vasp")
    # create dummy entries using the Li sites in the structure
    base_struct = struct.copy(sanitize=True)
    base_struct.remove_species(["Li"])
    base_entry = ComputedStructureEntry(structure=base_struct, energy=0, correction=0)
    inserted_entries = []

    sga = SpacegroupAnalyzer(struct)
    # Add a the same fake energy to each group of symmetrically equivalent sites
    # If the code works properly, three fake insertion_energy (inserted structure energy - base energy)/n_Li
    # should be decorated on the sites of the structure in the cep
    symm_struct = sga.get_symmetrized_structure()
    for gnum, site_group in enumerate(symm_struct.equivalent_sites):
        if site_group[0].species_string != "Li":
            continue
        for isite in site_group:
            tmp_struct_ = base_struct.copy()
            tmp_struct_.insert(0, "Li", isite.frac_coords, properties={"magmom": 0.0})
            new_entry = ComputedStructureEntry(
                structure=Structure.from_dict(tmp_struct_.as_dict()),
                energy=gnum,
                correction=0,
            )
            inserted_entries.append(new_entry)
    return {"base_entry": base_entry, "inserted_entries": inserted_entries}


@pytest.fixture(scope="session")
def test_get_cep_from_grouped_entries(entries):
    cep = get_cep_from_grouped_entries(
        base_entry=entries["base_entry"],
        inserted_entries=entries["inserted_entries"],
        migrating_species="Li",
    )
    assert {
        int(isite.properties["insertion_energy"]) for isite in cep.full_sites.sites
    } == {0, 1, 2}


def test_get_sc_fromstruct(entries):
    struct = entries["base_entry"].structure
    sc = get_sc_fromstruct(struct)
    assert np.linalg.matrix_rank(sc) == 3  # non singular
    sc = _get_sc_from_struct_ase(struct)
    assert np.linalg.matrix_rank(sc) == 3  # non singular
