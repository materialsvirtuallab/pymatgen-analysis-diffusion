# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

"""
Function to add edge data to MigrationGraph through 2 SC structures
"""

__author__ = "Haoming Li"
__copyright__ = "Copyright 2021, The Materials Project"
__maintainer__ = "Haoming Li"
__email__ = "HLi98@lbl.gov"
__date__ = "February 2, 2021"

import logging
from typing import Tuple
import numpy as np
from pymatgen.core.structure import Structure, PeriodicSite
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.diffusion.neb.full_path_mapper import MigrationGraph, MigrationHop
from pymatgen.analysis.diffusion.utils.parse_entries import get_matched_structure_mapping
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

logger = logging.getLogger(__name__)


def add_edge_data_from_sc(
    mg: MigrationGraph,
    i_sc: Structure,
    e_sc: Structure,
    data_array: list,
    key: str = "custom_key",
    use_host_sg: bool = True,
) -> None:
    """
    Add a data entry and key to edges within FullPathMapper object with the same hop_label.
    These hops are equivalent by symmetry to the 2 positions given in the supercell structures.

    Args:
        i_sc: Supercell structure containing working ion at initial position
        e_sc: Supercell structure containing working ion at ending position
        data_array: The data to be added to the edges
        key: Key of the edge attribute to be added
        use_host_sg: Flag t whether or not to use the host structure's spacegroup to initiate MigrationHop

    Returns:
        None
    """
    wi = list(mg.m_graph.graph.edges(data=True))[0][2]["hop"].isite.specie.name
    i_wi = [x for x in i_sc.sites if x.species_string == wi]
    e_wi = [x for x in e_sc.sites if x.species_string == wi]
    if len(i_wi) != 1 or len(e_wi) != 1:
        raise ValueError("The number of working ions in each supercell structure should be one")
    isite, esite = i_wi[0], e_wi[0]
    uhop_index, mh_from_sc = get_unique_hop(mg, i_sc, isite, esite, use_host_sg)
    add_dict = {key: data_array}
    mg.add_data_to_similar_edges(target_label=uhop_index, data=add_dict, m_hop=mh_from_sc)


def get_uc_pos(
    isite: PeriodicSite,
    esite: PeriodicSite,
    uc: Structure,
    sc: Structure,
    sm: StructureMatcher,
) -> Tuple[PeriodicSite, PeriodicSite, PeriodicSite]:
    """Take positions in the supercel and transform into the unitcell positions

    Args:
        isite: initial site in the SC
        esite: ending site in the SC
        uc: Unit Cell structre
        sc: Super Cell structure
        sm: StructureMatcher object with the working ion ignored

    Returns:
        The positions in the unit cell
    """
    mapping = get_matched_structure_mapping(base=uc, inserted=sc, sm=sm)
    if mapping is None:
        raise ValueError("Cannot obtain inverse mapping, consider lowering tolerances " "in StructureMatcher")
    sc_m, total_t = mapping
    sc_ipos = isite.frac_coords
    sc_ipos_t = sc_ipos - total_t
    uc_ipos = sc_ipos_t.dot(sc_m)
    image_trans = np.floor(uc_ipos)
    uc_ipos = uc_ipos - image_trans
    uc_ipos = _get_first_close_site(uc_ipos, uc)

    sc_epos = esite.frac_coords
    sc_epos_t = sc_epos - total_t
    uc_epos = sc_epos_t.dot(sc_m)
    uc_epos = uc_epos - image_trans
    uc_epos = _get_first_close_site(uc_epos, uc)

    sc_msite = PeriodicSite(
        esite.specie,
        (sc_ipos + sc_epos) / 2,
        esite.lattice,
    )
    sc_mpos = sc_msite.frac_coords

    sc_mpos_t = sc_mpos - total_t
    uc_mpos = sc_mpos_t.dot(sc_m)
    uc_mpos = uc_mpos - image_trans
    uc_mpos = _get_first_close_site(uc_mpos, uc)

    p0 = PeriodicSite(isite.specie, uc_ipos, uc.lattice)
    p1 = PeriodicSite(esite.specie, uc_mpos, uc.lattice)
    p2 = PeriodicSite(esite.specie, uc_epos, uc.lattice)
    return p0, p1, p2


def _get_first_close_site(frac_coord, structure, stol=0.1):
    for site in structure.sites:
        dist, image = structure.lattice.get_distance_and_image(frac_coord, site.frac_coords)
        if dist < stol:
            return np.add(site.frac_coords, image)
    return frac_coord


def mh_eq(mh1, mh2):
    """
    Allow for symmetric matching of MigrationPath objects with variable precession

    Args:
        mh1: MigrationHop object
        mh2: MigrationHop object

    Returns:
        Boolean True if they're equal, False if they are not
    """
    assert mh1.symm_structure == mh2.symm_structure
    return mh1.__eq__(mh2)


def get_unique_hop(
    mg: MigrationGraph,
    sc: Structure,
    isite: PeriodicSite,
    esite: PeriodicSite,
    use_host_sg: bool = True,
) -> Tuple[int, MigrationHop]:
    """Get the unique hop label that correspond to two end positions in the SC

    Args:
        mg: Object containing the migration analysis
        sc: Structure of the supercell used for the NEB calculation
        isite: Initial position in the supercell
        esite: Final position in the supercell
        use_host_sg: Flag t whether or not to use the host structure's spacegroup to initiate MigrationHop

    Returns:
        The index of the unique hop, the MigrationHop object trasformed from the SC
    """
    sm = StructureMatcher(ignored_species=[list(mg.m_graph.graph.edges(data=True))[0][2]["hop"].isite.specie.name])
    uc_isite, uc_msite, uc_esite = get_uc_pos(isite, esite, mg.symm_structure, sc, sm)
    if use_host_sg:
        base_ss = SpacegroupAnalyzer(mg.host_structure, symprec=mg.symprec).get_symmetrized_structure()
        mh_from_sc = MigrationHop(
            uc_isite, uc_esite, symm_structure=mg.symm_structure, host_symm_struct=base_ss, symprec=mg.symprec
        )
    else:
        mh_from_sc = MigrationHop(uc_isite, uc_esite, symm_structure=mg.symm_structure, symprec=mg.symprec)
    result = []
    for k, v in mg.unique_hops.items():
        # tolerance may be changed here
        if mh_eq(v["hop"], mh_from_sc):
            result.append(k)

    if len(result) > 1:
        raise ValueError("Too many matches between UC and SC")
    if len(result) == 0:
        raise ValueError("No matches between UC and SC")

    assert mg.symm_structure.spacegroup.are_symmetrically_equivalent([uc_msite], [mh_from_sc.msite])
    return result[0], mh_from_sc
