# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.
"""
Functions for querying Materials Project style MongoStores that contains
cathode materials The functions are isolated from the rest of the package so
that the rest of the package will not depend on Maggma
"""

import json
import logging
import zlib
from itertools import chain
from typing import Union

import gridfs
from maggma.stores import MongoStore
from monty.serialization import MontyDecoder
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry

__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@lbl.gov"
__date__ = "July 21, 2019"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
compat = MaterialsProjectCompatibility("Advanced")


def get_ent_from_db(
    elec_store: MongoStore,
    material_store: MongoStore,
    tasks_store: MongoStore,
    batt_id: Union[str, int] = None,
    task_id: Union[str, int] = None,
    get_aeccar: bool = False,
    working_ion: str = "Li",
    add_fields: list = None,
    get_initial: bool = False,
):
    """
    Get the migration path information in the form of a ComputedEntryGraph
    object from the an atomate data stack

    The algorithm gets all tasks with structures that are valid (i.e. matches a
    base structure) and generates a migration pathway object using all possible
    relaxed working ion positions found in that set. Since each material entry
    might contain multiple calculations with different cell sizes, this will
    have to work at the task level. Need to group tasks together based on the
    cell size of the base material

    Note that SPGlib is some times inconsistent when it comes to the getting
    the number of symmetry operations for a given structure. Sometimes
    structures that are the same using StructureMatcher.fit will have
    different number of symmetry operation. As such we will check the number
    of operations for each base structure in a given family of structures
    and take the case with the highest number symmetry operations In cases
    where AECCAR is required, only the tasks with AECCARs will have this data.

    Args:

        elec_store: Electrode documents one per each similar group of
            insertion materials, can also use any db that contains a
        material_ids list with topotactic structures
        material_store: Material documenets one per each similar structure (
            multiple tasks)
        tasks_store: Task documents one per each VASP calculation
        batt_id: battery id to lookup in a database.
        task_id: if battery id is not provided then look up a materials id.
        get_aeccar: If True, only find base tasks with the charge density stored
        working_ion: Name of the working ion. Defaults to 'Li'.
        add_fields: Take these fields from the task_documents and store them
            in ComputedStructureEntry
        get_initial: Store the initial structure of a calculation

    """

    task_ids_type = type(material_store.query_one({})["task_ids"][0])
    material_ids_type = type(elec_store.query_one({})["material_ids"][0])

    logger.debug(material_ids_type)

    def get_task_ids_from_batt_id(b_id):
        mat_ids = list(map(task_ids_type, elec_store.query_one({"battid": b_id})["material_ids"]))
        logger.debug(f"mat_ids : {mat_ids}")
        l_task_ids = [imat["task_ids"] for imat in material_store.query({"task_ids": {"$in": mat_ids}})]
        l_task_ids = list(chain.from_iterable(l_task_ids))
        logger.debug(f"l_task_ids : {l_task_ids}")
        return l_task_ids

    def get_batt_ids_from_task_id(t_id):
        l_task_ids = [c0["task_ids"] for c0 in material_store.query({"task_ids": {"$in": [int(t_id)]}})]
        l_task_ids = list(chain.from_iterable(l_task_ids))
        l_task_ids = list(map(material_ids_type, l_task_ids))
        logger.debug(f"l_task_ids : {l_task_ids}")
        l_mat_ids = [c0["material_ids"] for c0 in elec_store.query({"material_ids": {"$in": l_task_ids}})]
        l_mat_ids = list(chain.from_iterable(l_mat_ids))
        l_mat_ids = list(map(task_ids_type, l_mat_ids))
        logger.debug(f"l_mat_ids : {l_mat_ids}")
        l_task_ids = [c0["task_ids"] for c0 in material_store.query({"task_ids": {"$in": l_mat_ids}})]
        l_task_ids = list(chain.from_iterable(l_task_ids))
        logger.debug(f"l_task_ids : {l_task_ids}")
        return l_task_ids

    def get_entry(task_doc, base_with_aeccar=False, add_fields=None, get_initial=None):
        # we don't really need to think about compatibility for now if just
        # want to make a code that automate NEB calculations
        tmp_struct = Structure.from_dict(task_doc["output"]["structure"])
        settings_dict = dict(
            potcar_spec=task_doc["calcs_reversed"][0]["input"]["potcar_spec"],
            rung_type=task_doc["calcs_reversed"][0]["run_type"],
        )
        if "is_hubbard" in task_doc["calcs_reversed"][0].keys():
            settings_dict["hubbards"] = task_doc["calcs_reversed"][0]["hubbards"]
            settings_dict["is_hubbard"] = (task_doc["calcs_reversed"][0]["is_hubbard"],)

        entry = ComputedStructureEntry(
            structure=tmp_struct,
            energy=task_doc["output"]["energy"],
            parameters=settings_dict,
            entry_id=task_doc["task_id"],
        )
        if base_with_aeccar:
            logger.debug("test")
            aec_id = tasks_store.query_one({"task_id": entry.entry_id})["task_id"]
            aeccar = get_aeccar_from_store(tasks_store, aec_id)
            entry.data.update({"aeccar": aeccar})

        if add_fields:
            for field in add_fields:
                if field in task_doc:
                    entry.data.update({field: task_doc[field]})
        if get_initial:
            entry.data.update({"initial_structure": task_doc["input"]["structure"]})

        return entry

    # Require a single base entry and multiple inserted entries to populate
    # the migration pathways

    # getting a full list of task ids
    # Batt_id -> material_id -> all task_ids
    # task_id -> mat_ids -> Batt_ids -> material_id -> all task_ids
    if batt_id:
        all_tasks = get_task_ids_from_batt_id(batt_id)
    else:
        all_tasks = get_batt_ids_from_task_id(task_id)
    # get_all the structures
    if get_aeccar:
        all_ents_base = [
            get_entry(
                c0,
                base_with_aeccar=True,
                add_fields=add_fields,
                get_initial=get_initial,
            )
            for c0 in tasks_store.query(
                {
                    "task_id": {"$in": all_tasks},
                    "elements": {"$nin": [working_ion]},
                    "calcs_reversed.0.aeccar0_fs_id": {"$exists": 1},
                }
            )
        ]
    else:
        all_ents_base = [
            get_entry(c0)
            for c0 in tasks_store.query({"task_id": {"$in": all_tasks}, "elements": {"$nin": [working_ion]}})
        ]
    logger.debug(f"Number of base entries: {len(all_ents_base)}")

    all_ents_insert = [
        get_entry(c0, add_fields=add_fields, get_initial=get_initial)
        for c0 in tasks_store.query({"task_id": {"$in": all_tasks}, "elements": {"$in": [working_ion]}})
    ]
    logger.debug(f"Number of inserted entries: {len(all_ents_insert)}")
    tmp = [f"{itr.name}({itr.entry_id})" for itr in all_ents_insert]
    logger.debug(f"{tmp}")
    return all_ents_base, all_ents_insert


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
