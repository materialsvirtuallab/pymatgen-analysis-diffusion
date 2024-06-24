"""Functions for creating supercells for NEB calculations."""

from __future__ import annotations

import logging

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

# Helper functions for MigrationHop.get_sc_struture


def get_sc_fromstruct(
    base_struct: Structure,
    min_atoms: int = 80,
    max_atoms: int = 240,
    min_length: float = 10.0,
) -> list[list[int]]:
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
) -> list[list[int]] | None:
    """
    Generate the best supercell from a unitcell using the pymatgen CubicSupercellTransformation.

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


def get_start_end_structures(
    isite: PeriodicSite,
    esite: PeriodicSite,
    base_struct: Structure,
    sc_mat: list[list[int | float]],
    vac_mode: bool,
    debug: bool = False,
    tol: float = 1e-5,
) -> tuple[Structure, Structure, Structure]:
    """
    Obtain the starting and terminating structures in a supercell for NEB calculations.

    Args:
        isite: Initial site index.
        esite: End site index.
        hop: object presenting the migration event
        base_struct: unit cell representation of the structure
        sc_mat: supercell transformation to create the simulation cell for the NEB calc
        vac_mode: Vacuum mode.
        debug: debug mode.
        tol: toleranace for identifying isite/esite within base_struct.

    Returns:
        initial structure, final structure, empty structure all in the supercell
    """

    def remove_site_at_pos(structure: Structure, site: PeriodicSite, tol: float):
        new_struct_sites = []
        for isite in structure:
            if not vac_mode or (isite.distance(site) <= tol):
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
        start_struct = remove_site_at_pos(start_struct, e_ref_, tol)
        end_struct = remove_site_at_pos(end_struct, i_ref_, tol)
    return start_struct, end_struct, base_sc
