# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

__author__ = "Haoming Li"
__copyright__ = "Copyright 2021, The Materials Project"
__maintainer__ = "Haoming Li"
__email__ = "HLi98@lbl.gov"
__date__ = "February 2, 2021"

from pymatgen import Structure, PeriodicSite
import numpy as np
from typing import Tuple, Union
from pymatgen_diffusion.neb.full_path_mapper import FullPathMapper
from pymatgen_diffusion.neb.pathfinder import MigrationPath
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen_diffusion.utils.parse_entries import get_matched_structure_mapping


def add_edge_data_from_sc(fpm, i_sc, e_sc, data_array, key="custom_key"):
    """
    Add a data entry and key to edges within FullPathMapper object with the same hop_label.
    These hops are equivalent by symmetry to the 2 positions given in the supercell structures.

    Args:
        i_sc: Supercell structure containing working ion at initial position
        e_sc: Supercell structure containing working ion at ending position
        data_array: The data to be added to the edges
        key: Key of the edge attribute to be added
    """
    wi = fpm.migrating_specie.name
    i_wi = [x for x in i_sc.sites if x.species_string == wi]
    e_wi = [x for x in e_sc.sites if x.species_string == wi]
    if len(i_wi) != 1 or len(e_wi) != 1:
        raise ValueError(
            "The number of working ions in each supercell structure should be one"
        )
    isite, esite = i_wi[0], e_wi[0]
    uhop_index = get_unique_hop(fpm, i_sc, isite, esite)
    add_dict = {key: data_array}
    fpm.add_data_to_similar_edges(target_label=uhop_index, data=add_dict)


# the functions below are taken from repo cath_scripts by Jimmy Shen


def get_uc_pos(
    isite: PeriodicSite,
    esite: PeriodicSite,
    uc: Structure,
    sc: Structure,
    sm: StructureMatcher,
) -> Tuple[PeriodicSite]:
    """Take a position in the supercel and return it in the unitcell
    sc_site : Li site in the SC
    uc : Unit Cell structre
    sc : Super Cell structure
    """
    mapping = get_matched_structure_mapping(base=uc, inserted=sc, sm=sm)
    if mapping is None:
        raise ValueError(
            "Cannot obtain inverse mapping, consider lowering tolerances "
            "in StructureMatcher"
        )
    sc_m, total_t = mapping
    li_pos = isite.frac_coords
    li_pos = li_pos - total_t
    # get the translation that maps the initial position into the cell
    uc_ipos = li_pos.dot(sc_m)
    image_trans = np.floor(uc_ipos)
    uc_ipos = uc_ipos - image_trans
    uc_ipos = _get_first_close_site(uc_ipos, uc)

    li_pos = esite.frac_coords
    li_pos = li_pos - total_t
    uc_epos = li_pos.dot(sc_m)
    uc_epos = uc_epos - image_trans
    uc_epos = _get_first_close_site(uc_epos, uc)

    msite = PeriodicSite(
        esite.specie,
        (uc_ipos + uc_epos) / 2,
        esite.lattice,
    )
    li_pos = msite.frac_coords
    uc_mpos = li_pos

    p0 = PeriodicSite(isite.specie, uc_ipos, uc.lattice)
    p1 = PeriodicSite(esite.specie, uc_mpos, uc.lattice)
    p2 = PeriodicSite(esite.specie, uc_epos, uc.lattice)
    return p0, p1, p2


def _get_first_close_site(frac_coord, structure, stol=0.1):
    for site in structure.sites:
        dist, image = structure.lattice.get_distance_and_image(
            frac_coord, site.frac_coords
        )
        if dist < stol:
            return np.add(site.frac_coords, image)


def mg_eq(mg1, mg2, symm_prec=0.0001):
    """
    Allow for symmetric matching of MigrationPath objects with variable precession
    Args:
        mg1: MigrationPath object
        mg2: MigrationPath object
        symm_prec: tolerence

    Returns:

    """
    assert mg1.symm_structure == mg2.symm_structure
    if mg1.symm_structure.spacegroup.are_symmetrically_equivalent(
        (mg1.isite, mg1.msite, mg1.esite),
        (mg2.isite, mg2.msite, mg2.esite),
        symm_prec=symm_prec,
    ):
        return True
    return False


def get_unique_hop(
    fpm: FullPathMapper,
    sc: Structure,
    isite: PeriodicSite,
    esite: PeriodicSite,
) -> Union[str, int]:
    """Get the unique hop label that correspond to two end positions in the SC

    Args:
        fpm: Object containing the migration analysis
        sc: Structure of the supercell used for the NEB calculation
        isite: Initial position in the supercell
        esite: Final position in the supercell

    Returns:
        The index of the unique hop

    """
    sm = StructureMatcher(ignored_species=[fpm.migrating_specie])
    uc_isite, uc_msite, uc_esite = get_uc_pos(isite, esite, fpm.symm_structure, sc, sm)
    mg_from_sc = MigrationPath(uc_isite, uc_esite, symm_structure=fpm.symm_structure)
    result = []
    for k, v in fpm.unique_hops.items():
        # may be change the tolerance here.
        if mg_eq(v["hop"], mg_from_sc, symm_prec=0.05):
            result.append(k)

        # if v['hop'] == mg_from_sc:
        #     print('Matched old way')
        #     result.append(k)

    if len(result) > 1:
        raise ValueError("Too many matches between UC and SC")
    if len(result) == 0:
        raise ValueError("No matches between UC and SC")

    # makesure that the midpint is also the same
    assert fpm.symm_structure.spacegroup.are_symmetrically_equivalent(
        [uc_msite], [mg_from_sc.msite]
    )
    return result[0]
