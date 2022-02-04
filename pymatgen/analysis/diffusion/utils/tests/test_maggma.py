# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.
import os

import pytest
from maggma.stores import JSONStore

from pymatgen.analysis.diffusion.utils.maggma import get_entries_from_dbs

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/test_files"

__author__ = "Jimmy Shen"
__version__ = "1.0"
__date__ = "April 10, 2019"


@pytest.fixture
def maggma_stores():
    return {
        "sgroups": JSONStore(f"{dir_path}/maggma_sgroup_store.json", key="group_id"),
        "materials": JSONStore(f"{dir_path}/maggma_materials_store.json", key="material_id"),
    }


def test(maggma_stores):
    base_ents, inserted_ents = get_entries_from_dbs(
        maggma_stores["sgroups"], maggma_stores["materials"], "Mg", material_id="mvc-6910_Mg"
    )

    # check that the entries have been created
    def has_mg(ent):
        return "Mg" in ent.composition.as_dict().keys()

    assert all(map(has_mg, inserted_ents))
    assert not any(map(has_mg, base_ents))
