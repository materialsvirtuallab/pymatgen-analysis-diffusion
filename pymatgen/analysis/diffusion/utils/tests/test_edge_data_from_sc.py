# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

import os
import numpy as np
import pytest
from pymatgen.core.structure import Structure, PeriodicSite
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.diffusion.neb.full_path_mapper import MigrationGraph
from pymatgen.analysis.diffusion.utils.edge_data_from_sc import add_edge_data_from_sc, get_uc_pos, get_unique_hop

test_dir = os.path.dirname(os.path.realpath(__file__))

__author__ = "Haoming Li"
__version__ = "1.0"
__date__ = "February 17, 2021"

uc_full_sites = Structure.from_file(f"{test_dir}/test_files/Li4Sr3Fe2O7_uc.vasp")
input_struct_i = Structure.from_file(f"{test_dir}/test_files/Sr3Fe2O7_sc_i.vasp")
input_struct_e = Structure.from_file(f"{test_dir}/test_files/Sr3Fe2O7_sc_e.vasp")

mg_uc_full_sites = Structure.from_file(f"{test_dir}/test_files/Mg3VOPO4_uc.vasp")
mg_input_struct_i = Structure.from_file(f"{test_dir}/test_files/Mg3VOPO4_sc_i.vasp")
mg_input_struct_e = Structure.from_file(f"{test_dir}/test_files/Mg3VOPO4_sc_e.vasp")

mg_Li = MigrationGraph.with_distance(structure=uc_full_sites, migrating_specie="Li", max_distance=5)
mg_Mg = MigrationGraph.with_distance(structure=mg_uc_full_sites, migrating_specie="Mg", max_distance=4)


def test_add_edge_data_from_sc():
    errors = []

    test_key = "test_key"
    test_array = [0, 1, 2, 3, 4]
    add_edge_data_from_sc(
        mg_Li,
        i_sc=input_struct_i,
        e_sc=input_struct_e,
        data_array=test_array,
        key=test_key,
    )

    edge_data = []
    for u, v, d in mg_Li.m_graph.graph.edges(data=True):
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

    assert not errors, "errors occured:\n" + "\n".join(errors)


def test_get_uc_pos():
    errors = []

    # set up parameters to initiate get_uc_pos
    uc_lattice = mg_Li.symm_structure.lattice
    isite = [x for x in input_struct_i.sites if x.species_string == "Li"][0]
    esite = [x for x in input_struct_e.sites if x.species_string == "Li"][0]
    sm = StructureMatcher(ignored_species=[list(mg_Li.m_graph.graph.edges(data=True))[0][2]["hop"].isite.specie.name])
    wi_specie = mg_Li.symm_structure[-1].specie

    p0, p1, p2 = get_uc_pos(isite, esite, mg_Li.symm_structure, input_struct_i, sm)

    # generate correct sites to compare
    test_p0 = PeriodicSite(
        wi_specie, np.array([2.91418875, 1.02974425, 4.4933425]), uc_lattice, coords_are_cartesian=True
    )
    test_p1 = PeriodicSite(
        wi_specie, np.array([4.82950555, 1.0247028, 4.10369437]), uc_lattice, coords_are_cartesian=True
    )
    test_p2 = PeriodicSite(
        wi_specie, np.array([6.74482475, 1.01967025, 3.7140425]), uc_lattice, coords_are_cartesian=True
    )

    if not test_p0.__eq__(p0):
        errors.append("Initial site does not match")
    if not test_p1.__eq__(p1):
        errors.append("Middle site does not match")
    if not test_p2.__eq__(p2):
        errors.append("Ending site does not match")

    assert not errors, "errors occured:\n" + "\n".join(errors)


def test_get_unique_hop_host():
    results = get_unique_hop(mg_Mg, mg_input_struct_i, mg_input_struct_i[0], mg_input_struct_e[0], use_host_sg=True)
    assert results[0] == 2


def test_get_unique_host_nonhost():
    with pytest.raises(Exception) as exc_info:
        get_unique_hop(mg_Mg, mg_input_struct_i, mg_input_struct_i[0], mg_input_struct_e[0], use_host_sg=False)
    expected_exception = "No symmetrically equivalent site was found for [0.53593472 2.8352428  4.54752366] Mg"
    assert exc_info.value.args[0] == expected_exception
