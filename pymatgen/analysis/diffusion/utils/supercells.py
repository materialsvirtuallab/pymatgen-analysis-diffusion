# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.
"""
Functions for creating supercells for NEB calculations
"""
import logging
from typing import List, Tuple, Union, Optional

import numpy as np

# from ase.build import find_optimal_cell_shape, get_deviation_from_optimal_cell_shape
# from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import PeriodicSite, Structure
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation

__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2019, The Materials Project"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@lbl.gov"
__date__ = "Feb 11, 2021"

logger = logging.getLogger(__name__)

# Helper functions for MigraionHop.get_sc_struture


def get_sc_fromstruct(
    base_struct: Structure,
    min_atoms: int = 80,
    max_atoms: int = 240,
    min_length: float = 10.0,
) -> List[List[int]]:
    """
    Generate the best supercell from a unitcell.
    The CubicSupercellTransformation from PMG is much faster but don't iterate over as many
    supercell configurations so it's less able to find the best configuration in a give cell size.
    We try the PMG's cubic supercell transformation with a cap on the number of atoms (max_atoms).
    The min_length is decreased by 10% (geometrically) until a supercell can be constructed.

    Args:
        base_struct: structure of the unit cell
        max_atoms: Maximum number of atoms allowed in the supercell.
        min_atoms: Minimum number of atoms allowed in the supercell.
        min_length: Minimum length of the smallest supercell lattice vector.

    Returns:
        struc_sc: Supercell that is as close to cubic as possible
    """
    m_len = min_length
    struct_sc = None
    while struct_sc is None:
        struct_sc = _get_sc_from_struct_pmg(base_struct, min_atoms, max_atoms, m_len)
        max_atoms += 1
    return struct_sc


def _get_sc_from_struct_pmg(
    base_struct: Structure,
    min_atoms: int = 80,
    max_atoms: int = 240,
    min_length: float = 10.0,
) -> Optional[List[List[int]]]:
    """
    Generate the best supercell from a unitcell using the pymatgen CubicSupercellTransformation

    Args:
        base_struct: structure of the unit cell
        max_atoms: Maximum number of atoms allowed in the supercell.
        min_atoms: Minimum number of atoms allowed in the supercell.
        min_length: Minimum length of the smallest supercell lattice vector.

    Returns:
        3x3 matrix: supercell matrix

    """
    cst = CubicSupercellTransformation(min_atoms=min_atoms, max_atoms=max_atoms, min_length=min_length)

    try:
        cst.apply_transformation(base_struct)
    except BaseException:
        return None
    return cst.transformation_matrix


# Something is broken with how ASE generates supercells
# now many calls to `find_optimal_cell_shape` results in
# `Failed to find a transformation matrix`
# So remove this funtionality now in favour of decreasing the max_length
#
# def _get_sc_from_struct_ase(
#     base_struct: Structure, min_atoms: int=80, max_atoms: int=240
# ) -> List[List[int]]:
#     """generate the best supercell from a unitcell using ASE's method, is slower but more exhaustive
#
#     Args:
#         base_struct (pymatgen.Structure): unit cell
#         min_atoms (int, optional): Minimum number of atoms in the desired supercell. Defaults to 80.
#
#     Returns:
#         3x3 matrix: supercell matrix
#     """
#
#     num_cells_min = int(np.ceil(min_atoms / base_struct.num_sites))
#     num_cells_max = int(np.ceil(max_atoms / base_struct.num_sites))
#
#     atoms = AseAtomsAdaptor().get_atoms(base_struct)
#     res = []
#     for icell in range(num_cells_min, num_cells_max):
#         if icell % 2 != 0 and icell % 3 != 0:
#             continue  # cells with many factors are more likely to be square
#         logger.info(f"Getting cell shape {icell} x unit cell")
#         sc_mat = find_optimal_cell_shape(atoms.cell, icell, "sc")
#         if sc_mat is None:
#             continue
#         deviation = get_deviation_from_optimal_cell_shape(np.dot(sc_mat, atoms.cell))
#         dd = {"deviation": deviation, "P1": sc_mat}
#         # base_struct_sc.to('poscar', f'/Users/lik/Desktop/enpoints_{base_struct_sc.num_sites}.vasp')
#         res.append(dd)
#         if deviation < 0.3:
#             logger.debug("Found good cell")
#             return sc_mat
#
#     else:
#         best_case = min(res, key=lambda x: x["deviation"])
#     logger.warning(
#         f"Could not find case with deviation from cubic was less than 0.3 using the ASE cubic supercell finder \
#         \nThe best one had a deviation of {best_case['deviation']}"
#     )
#     return best_case["P1"]


def get_start_end_structures(
    isite: PeriodicSite,
    esite: PeriodicSite,
    base_struct: Structure,
    sc_mat: List[List[Union[int, float]]],
    vac_mode: bool,
    debug: bool = False,
) -> Tuple[Structure, Structure, Structure]:
    """
    Obtain the starting and terminating structures in a supercell for NEB calculations.

    Args:
        hop: object presenting the migration event
        base_struct: unit cell representation of the structure
        sc_mat: supercell transformation to create the simulation cell for the NEB calc

    Returns:
        initial structure, final structure, empty structure all in the supercell
    """

    def remove_site_at_pos(structure: Structure, site: PeriodicSite):
        new_struct_sites = []
        for isite in structure:
            if not vac_mode or (isite.distance(site) <= 1e-8):
                continue
            new_struct_sites.append(isite)
        return Structure.from_sites(new_struct_sites)

    base_sc = base_struct.copy() * sc_mat

    start_struct = base_struct.copy() * sc_mat
    end_struct = base_struct.copy() * sc_mat

    sc_mat_inv = np.linalg.inv(sc_mat)

    if not vac_mode:
        # insertion the endpoints
        start_struct.insert(
            0,
            esite.species_string,
            np.dot(isite.frac_coords, sc_mat_inv),
            properties={"magmom": 0},
        )
        end_struct.insert(
            0,
            esite.species_string,
            np.dot(esite.frac_coords, sc_mat_inv),
            properties={"magmom": 0},
        )
    else:
        # remove the other endpoint
        ipos_sc = np.dot(isite.frac_coords, sc_mat_inv)
        epos_sc = np.dot(esite.frac_coords, sc_mat_inv)
        if debug:
            icart = base_sc.lattice.get_cartesian_coords(ipos_sc)
            ecart = base_sc.lattice.get_cartesian_coords(epos_sc)
            assert abs(np.linalg.norm(icart - ecart) - np.linalg.norm(isite.coords - esite.coords)) < 1e-5
        i_ref_ = PeriodicSite(species=esite.species_string, coords=ipos_sc, lattice=base_sc.lattice)
        e_ref_ = PeriodicSite(species=esite.species_string, coords=epos_sc, lattice=base_sc.lattice)
        start_struct = remove_site_at_pos(start_struct, e_ref_)
        end_struct = remove_site_at_pos(end_struct, i_ref_)
    return start_struct, end_struct, base_sc
