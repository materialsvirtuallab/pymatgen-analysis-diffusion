# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

import os
from pymatgen import Structure
from pymatgen_diffusion.neb.utils.edge_data_from_sc import add_edge_data_from_sc
from pymatgen_diffusion.neb.full_path_mapper import MigrationGraph

test_dir = os.path.dirname(os.path.realpath(__file__))

__author__ = "Haoming Li"
__version__ = "1.0"
__date__ = "February 17, 2021"

uc_full_sites = Structure.from_file(f"{test_dir}/Li4Sr3Fe2O7_uc.vasp")
input_struct_i = Structure.from_file(f"{test_dir}/Sr3Fe2O7_sc_i.vasp")
input_struct_e = Structure.from_file(f"{test_dir}/Sr3Fe2O7_sc_e.vasp")


def test_add_edge_data_from_sc():
    errors = []

    mg = MigrationGraph.with_distance(
        structure=uc_full_sites, migrating_specie="Li", max_distance=5
    )
    test_key = "test_key"
    test_array = [0, 1, 2, 3, 4]
    add_edge_data_from_sc(
        mg,
        i_sc=input_struct_i,
        e_sc=input_struct_e,
        data_array=test_array,
        key=test_key,
    )

    edge_data = []
    for u, v, d in mg.migration_graph.graph.edges(data=True):
        edge_data.append(d)
    hop_labels = []
    for i in edge_data:
        for k, v in i.items():
            if k == test_key:
                hop_labels.append(i["hop_label"])

    if not hop_labels:
        errors.append("No data was added to any edge")
    if not all(i == hop_labels[0] for i in hop_labels):
        errors.append("Not all data are added to the same unique hop")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))
