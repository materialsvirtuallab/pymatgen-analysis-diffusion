# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

"""
Functions for combining many ComputedEntry objects into FullPathMapper objects.
"""

from pymatgen import Structure, Lattice, Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen_diffusion.neb.full_path_mapper import generic_groupby
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
import numpy as np
from itertools import chain
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import gridfs
import zlib
import json
import logging
from monty.serialization import MontyDecoder
from typing import Union, List

__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@lbl.gov"
__date__ = "July 21, 2019"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
compat = MaterialsProjectCompatibility("Advanced")


def process_ents(
    all_ents_base,
    all_ents_insert,
    symprec=0.01,
    ltol=0.2,
    stol=0.3,
    angle_tol=5,
    working_ion="Li",
    use_strict_tol=False,
):
    """
    Return grouped entries based on how many inserted entries were grouped together
    For the inserted structure map each on onto this base structure

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
        list: List of dictionaries that each contain {'base' :
        ComputedStructureEntry, 'inserted' : [ComputedStructureEntry]}

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
    # s1_supercell == True makes supercell using s1
    sc = s1 * sc_m
    #     sc.translate_sites(list(range(len(sc))), -total_t)
    sc.lattice = Lattice.from_parameters(
        *sc.lattice.abc, *sc.lattice.angles, vesta=True
    )
    return sc_m, total_t


def get_inserted_on_base(
    base_ent, inserted_ent, sm
) -> Union[None, List[ComputedStructureEntry]]:
    """
    Create a new entries with:
    - The exact atomic positions as the base
    - All of the ignored species inserted at corresponding positions in the
        base cell
    - The energy of each new structure is
        1/num_ignored * (E_inserted - k * E_base)
        where k is the supercell size in terms of the unit cell

    Args:
        base_ent:
        inserted_ent:
        sm:

    Returns:
        :

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


def process_ents_old(
    all_ents_base,
    all_ents_insert,
    symprec=0.01,
    ltol=0.2,
    stol=0.3,
    angle_tol=5,
    working_ion="Li",
    only_single_cat=False,
    use_strict_tol=False,
):
    """
    Group the entries for generating ComputedEntryPath objects  jk

    Args:
        all_ents_base: Full list of base entires
        all_ents_insert: Full list of inserted entires
        symprec:  symmetry parameter for SpacegroupAnalyzer
        ltol: Fractional length tolerance for StructureMatcher
        stol: Site tolerance for StructureMatcher
        angle_tol: Angle tolerance fro StructureMatcher and SpacegroupAnalyzer
        working_ion: String for the working ion
        only_single_cat: If True, only use single cation insertions so the
            site energy is more accurate
        use_strict_tol: halve the ltol and stol parameter for more strict
        matching.

    Returns:
        list: List of dictionaries that each contain
            {'base' : ComputedStructureEntry,
            'inserted' : [ComputedStructureEntry]}

    """

    sm_no_wion = StructureMatcher(
        comparator=ElementComparator(),
        primitive_cell=False,
        ignored_species=[working_ion],
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
    )

    sm_no_wion_strict = StructureMatcher(
        comparator=ElementComparator(),
        primitive_cell=False,
        ignored_species=[working_ion],
        ltol=ltol / 2.0,
        stol=stol / 2.0,
        angle_tol=angle_tol,
    )
    # grouping of inserted strutures with base structures
    all_sga = [
        SpacegroupAnalyzer(
            itr_base_ent.structure, symprec=symprec, angle_tolerance=angle_tol
        )
        for itr_base_ent in all_ents_base
    ]
    id_and_symm = [
        (ient.entry_id, len(all_sga[itr_ent].get_space_group_operations()))
        for itr_ent, ient in enumerate(all_ents_base)
    ]
    logger.debug(
        f"The number of symmetry operations for each material is: \
            {id_and_symm}"
    )
    base_struct_labels = generic_groupby(
        all_ents_base, lambda x, y: sm_no_wion.fit(x.structure, y.structure)
    )
    # get best structure
    grouped_entries = []
    # for each similar group of base structures get the entry with the most
    # number of symmetry operations
    # Most of the time this will not be neccessary but structure matcher does
    # not work exactly the same way as SpaceGroupAnalyzer
    for itr in np.unique(base_struct_labels):
        indices_base = [i for i, x in enumerate(base_struct_labels) if x == itr]
        best_base_index = sorted(
            indices_base,
            key=lambda index_base: len(
                all_sga[index_base].get_space_group_operations()
            ),
        )[-1]
        grouped_entries.append(dict(base=all_ents_base[best_base_index], inserted=[]))

    # insert_struct_labels = np.unique(
    #     generic_groupby(all_ents_insert, lambda x, y: sm_no_wion.fit(
    #         x.structure, y.structure)))
    # return  all_ents_base[best_base_index],  all_ents_insert
    for insert_ent in all_ents_insert:
        # print(insert_ent.entry_id)
        if (
            only_single_cat
            and insert_ent.structure.composition.as_dict()[working_ion] > 1
        ):
            # print("too many")
            continue
        for itr_dict in grouped_entries:
            if use_strict_tol:
                sm_sites = sm_no_wion_strict
            else:
                sm_sites = sm_no_wion
            # print(sm_sites.ltol, sm_sites.stol)
            if sm_sites.fit(insert_ent.structure, itr_dict["base"].structure):
                itr_dict["inserted"].append(insert_ent)
            else:
                logger.debug(
                    f"Inserted material {insert_ent.entry_id} does not match with the base  {itr_dict['base'].entry_id}"
                )

    return grouped_entries


def get_aeccar_from_store(tstore, task_id):
    """
    Read the AECCAR grid_fs data into a Chgcar object

    Args:
        tstore (MongoStore): MongoStore for the tasks database
        task_id: The task_id of the material entry

    Returns:
        pymatgen Chrgcar object: The AECCAR data from a given task
    """
    m_task = tstore.query_one({"task_id": task_id})
    try:
        fs_id = m_task["calcs_reversed"][0]["aeccar0_fs_id"]
    except BaseException:
        logger.info("AECCAR0 Missing from task # {}".format(task_id))
        return None

    fs = gridfs.GridFS(tstore._collection.database, "aeccar0_fs")
    chgcar_json = zlib.decompress(fs.get(fs_id).read())
    aeccar0 = json.loads(chgcar_json, cls=MontyDecoder)

    try:
        fs_id = m_task["calcs_reversed"][0]["aeccar2_fs_id"]
    except BaseException:
        logger.info("AECCAR2 Missing from task # {}".format(task_id))
        return None

    fs = gridfs.GridFS(tstore._collection.database, "aeccar2_fs")
    chgcar_json = zlib.decompress(fs.get(fs_id).read())
    aeccar2 = json.loads(chgcar_json, cls=MontyDecoder)
    return aeccar0 + aeccar2
