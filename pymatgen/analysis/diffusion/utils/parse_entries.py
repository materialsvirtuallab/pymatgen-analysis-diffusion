# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

"""
Functions for combining many ComputedEntry objects into MigrationGraph objects.
"""
import logging
from typing import Dict, List, Optional, Union

import numpy as np
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2019, The Materials Project"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@lbl.gov"
__date__ = "July 21, 2019"

# Magic Numbers
# Eliminate cation sites that are too close to the sites in the base structure
BASE_COLLISION_R = 1.0
# Merge cation sites that are too close together
SITE_MERGE_R = 1.0

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_entries(
    base_entries: List[ComputedStructureEntry],
    inserted_entries: List[ComputedStructureEntry],
    migrating_ion_entry: ComputedEntry,
    symprec: float = 0.01,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> List[Dict]:
    """
    Process a list of base entries and inserted entries to create input for migration path analysis Each inserted
    entries can be mapped to more than one base entry. Return groups of structures decorated with the working ions
    to indicate the metastable sites, ranked by the number of working ion sites (highest number is the first).

    Args:
        base_entries: Full list of base entires
        inserted_entries: Full list of inserted entires
        migrating_ion_entry: The metallic phase of the working ion, used to calculate insertion energies.
        symprec:  symmetry parameter for SpacegroupAnalyzer
        ltol: Fractional length tolerance for StructureMatcher
        stol: Site tolerance for StructureMatcher
        angle_tol: Angle tolerance fro StructureMatcher and SpacegroupAnalyzer
        only_single_cat: If True, only use single cation insertions so the
            site energy is more accurate use_strict_tol: halve the ltol and
            stol parameter for more strict matching.

    Returns:
        list: List of dictionaries that each contain
        {'base' : Structure Object of host, 'inserted' : Structure object of all inserted sites}
    """
    working_ion = str(migrating_ion_entry.composition.elements[0])
    sm_no_wion = StructureMatcher(
        comparator=ElementComparator(),
        primitive_cell=False,
        ignored_species=[working_ion],
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
    )

    # grouping of inserted structures with base structures
    all_sga = [
        SpacegroupAnalyzer(itr_base_ent.structure, symprec=symprec, angle_tolerance=angle_tol)
        for itr_base_ent in base_entries
    ]

    entries_with_num_symmetry_ops = [
        (ient, len(all_sga[itr_ent].get_space_group_operations())) for itr_ent, ient in enumerate(base_entries)
    ]

    entries_with_num_symmetry_ops = sorted(entries_with_num_symmetry_ops, key=lambda x: x[0].energy_per_atom)
    entries_with_num_symmetry_ops = sorted(entries_with_num_symmetry_ops, key=lambda x: x[1], reverse=True)
    entries_with_num_symmetry_ops = sorted(
        entries_with_num_symmetry_ops,
        key=lambda x: x[0].structure.num_sites,
        reverse=True,
    )

    results = []

    def _meta_stable_sites(base_ent, inserted_ent):
        mapped_struct = get_inserted_on_base(
            base_ent,
            inserted_ent,
            migrating_ion_entry=migrating_ion_entry,
            sm=sm_no_wion,
        )
        if mapped_struct is None:
            return []
        return mapped_struct.sites

    for base_ent, _ in entries_with_num_symmetry_ops:
        # structure where the
        mapped_cell = base_ent.structure.copy()
        for j_inserted in inserted_entries:
            inserted_sites_ = _meta_stable_sites(base_ent, j_inserted)
            mapped_cell.sites.extend(inserted_sites_)

        struct_wo_sym_ops = _filter_and_merge(mapped_cell.get_sorted_structure())
        if struct_wo_sym_ops is None:
            logger.warning(
                f"No meta-stable sites were found during symmetry mapping for base {base_ent.entry_id}."
                "Consider playing with the various tolerances (ltol, stol, angle_tol)."
            )
            continue

        struct_sym = get_sym_migration_ion_sites(base_ent.structure, struct_wo_sym_ops, working_ion)
        results.append(
            {
                "base": base_ent.structure,
                "inserted": struct_sym,
            }
        )

    results = filter(lambda x: len(x["inserted"]) != 0, results)  # type: ignore
    results = sorted(results, key=lambda x: x["inserted"].composition[working_ion], reverse=True)
    return results


def get_matched_structure_mapping(base: Structure, inserted: Structure, sm: StructureMatcher):
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
        val, dist, sc_m, total_t, mapping = sm._strict_match(s1, s2, fu=fu, s1_supercell=True)
    except TypeError:
        return None
    sc = s1 * sc_m
    sc.lattice = Lattice.from_parameters(*sc.lattice.abc, *sc.lattice.angles, vesta=True)  # type: ignore
    return sc_m, total_t


def get_inserted_on_base(
    base_ent: ComputedStructureEntry,
    inserted_ent: ComputedStructureEntry,
    migrating_ion_entry: ComputedEntry,
    sm: StructureMatcher,
) -> Optional[Structure]:
    """
    For a structured-matched pair of base and inserted entries, map all of the Li positions in the inserted entry to
    positions in the base entry and return a new structure where all the sites are decorated with the insertion
    energy. Since the calculation of the insertion energy needs the energy of the metallic working ion,
    a `migrating_ion_entry` must also be provided.

    Args:
        base_ent: The entry for the host structure
        inserted_ent: The entry for the inserted structure
        migrating_ion_entry: The entry containing the migrating ion
        sm: StructureMatcher object used to obtain the mapping

    Returns:
        List of entries for each working ion in the list of
    """
    mapped_result = get_matched_structure_mapping(base_ent.structure, inserted_ent.structure, sm)
    if mapped_result is None:
        return None

    sc_m, total_t = mapped_result
    insertion_energy = get_insertion_energy(base_ent, inserted_ent, migrating_ion_entry)

    new_struct = base_ent.structure.copy()
    for ii, isite in enumerate(inserted_ent.structure.sites):
        if isite.species_string not in sm._ignored_species:
            continue
        li_pos = isite.frac_coords
        li_pos = li_pos + total_t
        li_uc_pos = li_pos.dot(sc_m)
        new_struct.insert(
            0,
            isite.species_string,
            li_uc_pos,
            properties={"insertion_energy": insertion_energy, "magmom": 0.0},
        )
    return new_struct


def get_sym_migration_ion_sites(
    base_struct: Structure,
    inserted_struct: Structure,
    migrating_ion: str,
    symprec: float = 0.01,
    angle_tol: float = 5.0,
) -> Structure:
    """
    Take one inserted entry then map out all symmetry equivalent copies of the cation sites in base entry.
    Each site is decorated with the insertion energy calculated from the base and inserted entries.

    Args:
        inserted_entry: entry that contains cation
        base_struct_entry: the entry containing the base structure
        migrating_ion_entry: the name of the migrating species
        symprec: the symprec tolerance for the space group analysis
        angle_tol: the angle tolerance for the space group analysis

    Returns:
        Structure with only the migrating ion sites decorated with insertion energies.
    """
    wi_ = migrating_ion

    sa = SpacegroupAnalyzer(base_struct, symprec=symprec, angle_tolerance=angle_tol)
    # start with the base structure but empty
    sym_migration_ion_sites = list(
        filter(
            lambda isite: isite.species_string == wi_,
            inserted_struct.sites,
        )
    )

    sym_migration_struct = Structure.from_sites(sym_migration_ion_sites)
    for op in sa.get_space_group_operations():
        struct_tmp = sym_migration_struct.copy()
        struct_tmp.apply_operation(symmop=op, fractional=True)
        for isite in struct_tmp.sites:
            if isite.species_string == wi_:
                sym_migration_struct.insert(
                    0,
                    wi_,
                    coords=np.mod(isite.frac_coords, 1.0),
                    properties=isite.properties,
                )

            # must clean up as you go or the number of sites explodes
            if len(sym_migration_struct) > 1:
                sym_migration_struct.merge_sites(tol=SITE_MERGE_R, mode="average")  # keeps removing duplicates
    return sym_migration_struct


def _filter_and_merge(inserted_structure: Structure) -> Union[Structure, None]:
    """
    For each site in a structure, split it into a migration sublattice where all sites contain the "insertion_energy"
    property and a host lattice. For each site in the migration sublattice if there is collision with the host sites,
    remove the migration site. Finally merge all the migration sites.
    """
    migration_sites = []
    base_sites = []
    for i_site in inserted_structure:
        if "insertion_energy" in i_site.properties and isinstance(i_site.properties["insertion_energy"], float):
            migration_sites.append(i_site)
        else:
            base_sites.append(i_site)
    if len(migration_sites) == 0:
        return None
    migration = Structure.from_sites(migration_sites)
    base = Structure.from_sites(base_sites)

    non_colliding_sites = []
    for i_site in migration.sites:
        col_sites = base.get_sites_in_sphere(i_site.coords, BASE_COLLISION_R)
        if len(col_sites) == 0:
            non_colliding_sites.append(i_site)
    res = Structure.from_sites(non_colliding_sites + base.sites)
    res.merge_sites(tol=SITE_MERGE_R, mode="average")
    return res


def get_insertion_energy(
    base_entry: ComputedStructureEntry,
    inserted_entry: ComputedStructureEntry,
    migrating_ion_entry: ComputedEntry,
) -> float:
    """
    Calculate the insertion energy for a given inserted entry
    Args:
        base_entry: The entry for the host structure
        inserted_entry: The entry for the inserted structure
        migrating_ion_entry: The entry for the metallic phase of the working ion
    Returns:
        The insertion energy defined as (E[inserted] - (E[Base] + n * E[working_ion]))/(n)
        Where n is the number of working ions and E[inserted].
        Additionally, and E[base] and E[inserted] are for structures of the same size (sans working ion)
    """
    wi_ = str(migrating_ion_entry.composition.elements[0])
    comp_inserted_no_wi = inserted_entry.composition.as_dict()
    comp_inserted_no_wi.pop(wi_)
    comp_inserted_no_wi = Composition.from_dict(comp_inserted_no_wi)
    _, factor_inserted = comp_inserted_no_wi.get_reduced_composition_and_factor()
    _, factor_base = base_entry.composition.get_reduced_composition_and_factor()
    e_base = base_entry.energy * factor_inserted / factor_base
    e_insert = inserted_entry.energy
    e_wi = migrating_ion_entry.energy_per_atom
    n_wi = inserted_entry.composition[wi_]

    return (e_insert - (e_base + n_wi * e_wi)) / n_wi
