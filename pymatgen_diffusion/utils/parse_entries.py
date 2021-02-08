# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

"""
Functions for combining many ComputedEntry objects into MigrationGraph objects.
"""

from pymatgen import Structure, Lattice, Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from itertools import chain
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import logging
from typing import Union, List, Dict

__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@lbl.gov"
__date__ = "July 21, 2019"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_ents(
    all_ents_base: List,
    all_ents_insert: List,
    symprec: float = 0.01,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
    working_ion: str = "Li",
    use_strict_tol: bool = False,
) -> List[Dict]:
    """
    Process a list of base entries and inserted entries to create input for migration path analysis
    Each inserted entries can be mapped to more than one base entry.
    Return groups of inserted entries based ranked by the number of inserted entries in each group

    Args:
        all_ents_base: Full list of base entires
        all_ents_insert: Full list of inserted entires
        symprec:  symmetry parameter for SpacegroupAnalyzer
        ltol: Fractional length tolerance for StructureMatcher
        stol: Site tolerance for StructureMatcher
        angle_tol: Angle tolerance fro StructureMatcher and SpacegroupAnalyzer
        working_ion: String for the working ion
        only_single_cat: If True, only use single cation insertions so the
            site energy is more accurate use_strict_tol: halve the ltol and
            stol parameter for more strict matching.

    Returns:
        list: List of dictionaries that each contain
        {'base' : ComputedStructureEntry, 'inserted' : [ComputedStructureEntry]}
    """

    if use_strict_tol:
        lt_val = ltol / 2
        st_val = stol / 2
    else:
        lt_val = ltol
        st_val = stol

    sm_no_wion = StructureMatcher(
        comparator=ElementComparator(),
        primitive_cell=False,
        ignored_species=[working_ion],
        ltol=lt_val,
        stol=st_val,
        angle_tol=angle_tol,
    )

    # grouping of inserted structures with base structures
    all_sga = [
        SpacegroupAnalyzer(
            itr_base_ent.structure, symprec=symprec, angle_tolerance=angle_tol
        )
        for itr_base_ent in all_ents_base
    ]

    entries_with_num_symmetry_ops = [
        (ient, len(all_sga[itr_ent].get_space_group_operations()))
        for itr_ent, ient in enumerate(all_ents_base)
    ]

    entries_with_num_symmetry_ops = sorted(
        entries_with_num_symmetry_ops, key=lambda x: x[0].energy_per_atom
    )
    entries_with_num_symmetry_ops = sorted(
        entries_with_num_symmetry_ops, key=lambda x: x[1], reverse=True
    )
    entries_with_num_symmetry_ops = sorted(
        entries_with_num_symmetry_ops,
        key=lambda x: x[0].structure.num_sites,
        reverse=True,
    )

    results = []
    for base_ent, _ in entries_with_num_symmetry_ops:
        mapped_inserted = [
            get_inserted_on_base(base_ent, j_inserted, sm_no_wion)
            for j_inserted in all_ents_insert
        ]
        results.append(
            {
                "base": base_ent,
                "inserted": [*chain.from_iterable(filter(None, mapped_inserted))],
            }
        )

    results = filter(lambda x: len(x["inserted"]) != 0, results)
    results = sorted(results, key=lambda x: len(x["inserted"]), reverse=True)
    return results


def get_matched_structure_mapping(
    base: Structure, inserted: Structure, sm: StructureMatcher
):
    """
    Get the mapping from the inserted structure onto the base structure,
    assuming that the inserted structure sans the working ion is some kind
    of SC of the base.

    Args:
        base: host structure, smaller cell
        inserted: bigger cell
        sm: StructureMatcher instance

    Returns:
        sc_m : supercell matrix to apply to s1 to get s2
        total-t : translation to apply on s1 * sc_m to get s2
    """
    s1, s2 = sm._process_species([base, inserted])
    fu, _ = sm._get_supercell_size(s1, s2)
    try:
        val, dist, sc_m, total_t, mapping = sm._strict_match(
            s1, s2, fu=fu, s1_supercell=True
        )
    except TypeError:
        return None
    sc = s1 * sc_m
    sc.lattice = Lattice.from_parameters(
        *sc.lattice.abc, *sc.lattice.angles, vesta=True
    )
    return sc_m, total_t


def get_inserted_on_base(
    base_ent: ComputedStructureEntry,
    inserted_ent: ComputedStructureEntry,
    sm: StructureMatcher,
) -> Union[None, List[ComputedStructureEntry]]:
    """
    For a structured-matched pair of base and inserted entries, map all of the
    Li positions in the inserted entry to positions in the base entry and return a new computed entry
    Args:
        base_ent: The entry for the host structure
        inserted_ent: The entry for the inserted structure
        sm: StructureMatcher object used to obtain the mapping

    Returns:
        List of entries for each working ion in the list of

    """
    mapped_result = get_matched_structure_mapping(
        base_ent.structure, inserted_ent.structure, sm
    )
    if mapped_result is None:
        return None
    else:
        sc_m, total_t = mapped_result
    res = []
    cc = base_ent.composition
    _, factor_base = cc.get_reduced_composition_and_factor()

    dd = inserted_ent.composition.as_dict()
    [dd.pop(ig_sp) for ig_sp in sm._ignored_species]
    _, factor_inserted = Composition(dd).get_reduced_composition_and_factor()

    total_sites = sum([inserted_ent.composition[j] for j in sm._ignored_species])
    res_en = (
        inserted_ent.energy - (base_ent.energy * (factor_inserted / factor_base))
    ) / total_sites + base_ent.energy

    for ii, isite in enumerate(inserted_ent.structure.sites):
        if isite.species_string not in sm._ignored_species:
            continue
        new_struct = base_ent.structure.copy()
        li_pos = isite.frac_coords
        li_pos = li_pos + total_t
        li_uc_pos = li_pos.dot(sc_m)
        new_struct.insert(0, isite.species_string, li_uc_pos)
        new_entry = ComputedStructureEntry(
            structure=new_struct,
            energy=res_en,
            parameters=inserted_ent.parameters,
            entry_id=str(inserted_ent.entry_id) + "_" + str(ii),
        )
        res.append(new_entry)

    return res
