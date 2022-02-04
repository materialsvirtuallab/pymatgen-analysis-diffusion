# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.
"""
Functions for querying Materials Project style MongoStores that contains
cathode materials The functions are isolated from the rest of the package so
that the rest of the package will not depend on Maggma
"""

__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@lbl.gov"
__date__ = "July 21, 2019"

import logging
from typing import Union

from maggma.stores import MongoStore
from monty.serialization import MontyDecoder
from pymatgen.entries.computed_entries import ComputedEntry

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_entries_from_dbs(
    structure_group_store: MongoStore, material_store: MongoStore, migrating_ion: str, material_id: Union[str, int]
):
    """
    Get the entries needed to construct a migration from a database that
    contains topotactically matched structures.

    Args:
        structure_group_store: Electrode documents one per each similar group of
            insertion materials, can also use any db that contains a
        material_store: Material documenets one per each similar structure (
            multiple tasks)
        migrating_ion: The name of the migrating ion
        material_ids list with topotactic structures
    """

    with structure_group_store as store:
        sg_doc = store.query_one({structure_group_store.key: material_id})
    ignored_species = migrating_ion
    base_entries = []
    inserted_entries = []
    with material_store as store:
        for m_doc in store.query({"material_id": {"$in": sg_doc["material_ids"]}}):
            if "GGA+U" in m_doc["entries"]:
                entry = MontyDecoder().process_decoded(m_doc["entries"]["GGA+U"])  # type: ComputedEntry
            elif "GGA" in m_doc["entries"]:
                entry = MontyDecoder().process_decoded(m_doc["entries"]["GGA"])
            else:
                raise RuntimeError("Missing GGA or GGA+U calc type in <entries>")

            if ignored_species in entry.composition.as_dict().keys():
                inserted_entries.append(entry)
            else:
                base_entries.append(entry)

    return base_entries, inserted_entries
